package com.striim.tranad;

import com.webaction.anno.AdapterType;
import com.webaction.anno.PropertyTemplate;
import com.webaction.anno.PropertyTemplateProperty;
import com.webaction.runtime.components.openprocessor.StriimOpenProcessor;
import com.webaction.runtime.containers.WAEvent;
import com.webaction.runtime.containers.IBatch;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.Map;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

@PropertyTemplate(
    name = "TranADScorer",
    type = AdapterType.process,
    properties = {
        @PropertyTemplateProperty(name = "apiEndpoint", type = String.class,
            required = false, defaultValue = "http://localhost:8000/score"),
        @PropertyTemplateProperty(name = "storeId", type = Integer.class,
            required = false, defaultValue = "1"),
        @PropertyTemplateProperty(name = "deviceId", type = Integer.class,
            required = false, defaultValue = "1"),
        @PropertyTemplateProperty(name = "timeoutMs", type = Integer.class,
            required = false, defaultValue = "5000")
    },
    outputType = com.webaction.proc.events.WAEvent.class,
    inputType  = com.webaction.proc.events.WAEvent.class
)
public class TranADScorer extends StriimOpenProcessor {

    private static final Logger logger = LogManager.getLogger(TranADScorer.class);
    private static final int WINDOW_SIZE = 10;
    private static final int N_FEATURES = 38;

    private HttpClient httpClient;
    private String apiEndpoint;
    private int storeId;
    private int deviceId;
    private int timeoutMs;

    // Internal sliding window buffer
    private final ArrayList<double[]> featureBuffer = new ArrayList<>(20);
    private final ArrayList<Integer> indexBuffer = new ArrayList<>(20);

    // Counters for observability
    private long rowsReceived = 0;
    private long windowsScored = 0;
    private long anomaliesDetected = 0;

    @Override
    public void run() {
        if (httpClient == null) {
            int effectiveTimeout = (timeoutMs > 0) ? timeoutMs : 5000;
            this.timeoutMs = effectiveTimeout;
            if (storeId <= 0) storeId = 1;
            if (deviceId <= 0) deviceId = 1;
            httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofMillis(effectiveTimeout))
                .build();
            if (apiEndpoint == null || apiEndpoint.isEmpty()) {
                apiEndpoint = "http://localhost:8000/score";
            }
            logger.info("TranADScorer initialized: endpoint={}, store={}, device={}, timeout={}ms",
                apiEndpoint, storeId, deviceId, effectiveTimeout);
        }

        IBatch<WAEvent> batch = getAdded();
        if (batch == null) return;

        Iterator<WAEvent> it = batch.iterator();
        while (it.hasNext()) {
            WAEvent outerEvent = it.next();

            try {
                // Extract raw CSV fields from the inner WAEvent
                Object innerObj = outerEvent.data;
                Object[] fields;

                if (innerObj != null &&
                    innerObj.getClass().getName().contains("WAEvent")) {
                    java.lang.reflect.Field dataField = innerObj.getClass().getField("data");
                    fields = (Object[]) dataField.get(innerObj);
                } else if (innerObj instanceof Object[]) {
                    fields = (Object[]) innerObj;
                } else {
                    logger.warn("Unexpected event type: {}",
                        innerObj != null ? innerObj.getClass().getName() : "null");
                    continue;
                }

                // DSVParser output: data[0..37] = 38 features (raw .txt, no header/index)
                double[] features = new double[N_FEATURES];
                for (int i = 0; i < N_FEATURES; i++) {
                    features[i] = Double.parseDouble(String.valueOf(fields[i]).trim());
                }

                featureBuffer.add(features);
                int rowIdx = (int) rowsReceived;
                indexBuffer.add(rowIdx);
                rowsReceived++;

                if (rowsReceived % 1000 == 0) {
                    logger.info("Progress: rows={}, scored={}, anomalies={}",
                        rowsReceived, windowsScored, anomaliesDetected);
                }

                // Score when buffer reaches window size (non-overlapping windows)
                if (featureBuffer.size() >= WINDOW_SIZE) {
                    // Build 2D array: [[38 floats], [38 floats], ...]
                    StringBuilder dataJson = new StringBuilder("[");
                    for (int row = 0; row < WINDOW_SIZE; row++) {
                        if (row > 0) dataJson.append(",");
                        dataJson.append("[");
                        double[] rowData = featureBuffer.get(row);
                        for (int col = 0; col < N_FEATURES; col++) {
                            if (col > 0) dataJson.append(",");
                            dataJson.append(rowData[col]);
                        }
                        dataJson.append("]");
                    }
                    dataJson.append("]");

                    JsonObject body = new JsonObject();
                    body.addProperty("store_id", storeId);
                    body.addProperty("device_id", deviceId);
                    body.add("data", JsonParser.parseString(dataJson.toString()));
                    body.addProperty("include_attribution", true);
                    body.addProperty("scoring_mode", "phase2_only");

                    String responseBody = callApiWithRetry(body.toString());

                    if (responseBody == null) {
                        logger.error("All retries exhausted for window starting at idx={}",
                            indexBuffer.get(0));
                    } else {
                        JsonObject resp = JsonParser.parseString(responseBody).getAsJsonObject();

                        int nAnomalies = resp.get("n_anomalies").getAsInt();
                        double threshold = resp.get("threshold").getAsDouble();
                        double anomalyRatio = resp.get("anomaly_ratio").getAsDouble();

                        String isAnomaly = (nAnomalies > 0) ? "true" : "false";

                        // Get top attributed dimension from first segment (if anomalous)
                        String topDimension = "";
                        String topElevation = "";
                        if (nAnomalies > 0) {
                            anomaliesDetected++;
                            JsonArray segments = resp.getAsJsonArray("anomaly_segments");
                            if (segments != null && segments.size() > 0) {
                                JsonObject seg = segments.get(0).getAsJsonObject();
                                JsonArray dims = seg.getAsJsonArray("attributed_dimensions");
                                if (dims != null && dims.size() > 0) {
                                    JsonObject topDim = dims.get(0).getAsJsonObject();
                                    topDimension = topDim.get("label").getAsString();
                                    topElevation = String.valueOf(topDim.get("mean_elevation").getAsDouble());
                                }
                            }
                        }

                        windowsScored++;

                        logger.info("Scored window idx=[{}, {}] anomaly={} n_anomalies={} threshold={}",
                            indexBuffer.get(0), indexBuffer.get(WINDOW_SIZE - 1),
                            isAnomaly, nAnomalies, threshold);

                        // Emit result via in-place WAEvent modification
                        Object[] resultData = new Object[] {
                            isAnomaly,                                          // data[0]
                            String.valueOf(nAnomalies),                         // data[1]
                            String.valueOf(threshold),                          // data[2]
                            String.valueOf(anomalyRatio),                       // data[3]
                            String.valueOf(indexBuffer.get(0)),                  // data[4] window start
                            String.valueOf(indexBuffer.get(WINDOW_SIZE - 1)),    // data[5] window end
                            topDimension,                                       // data[6]
                            topElevation                                        // data[7]
                        };

                        if (innerObj != null &&
                            innerObj.getClass().getName().contains("WAEvent")) {
                            java.lang.reflect.Field dataField = innerObj.getClass().getField("data");
                            dataField.set(innerObj, resultData);
                            send(innerObj);
                        } else {
                            send(resultData);
                        }
                    }

                    // Non-overlapping: clear entire buffer after scoring
                    featureBuffer.clear();
                    indexBuffer.clear();
                }

            } catch (Exception e) {
                logger.error("Error processing event: {}", e.getMessage(), e);
            }
        }
    }

    private String callApiWithRetry(String jsonBody) {
        int retries = 3;
        int attempt = 0;
        long backoffMs = 500;
        while (attempt <= retries) {
            try {
                HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(apiEndpoint))
                    .timeout(Duration.ofMillis(timeoutMs))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(jsonBody))
                    .build();
                HttpResponse<String> response = httpClient.send(
                    request, HttpResponse.BodyHandlers.ofString());
                int status = response.statusCode();
                if (status >= 200 && status < 300) return response.body();
                if (status >= 400 && status < 500) {
                    logger.error("Client error: status={}, body={}", status, response.body());
                    return null;
                }
                logger.warn("Server error: status={}, attempt={}/{}", status, attempt, retries);
            } catch (Exception e) {
                logger.warn("API exception: {}, attempt={}/{}", e.getMessage(), attempt, retries);
            }
            try { Thread.sleep(backoffMs); } catch (InterruptedException ie) {
                Thread.currentThread().interrupt(); return null;
            }
            backoffMs *= 2;
            attempt++;
        }
        return null;
    }

    @Override
    public void close() throws Exception {
        logger.info("TranADScorer shutting down. Final: rows={}, scored={}, anomalies={}",
            rowsReceived, windowsScored, anomaliesDetected);
    }

    @Override
    public Map getAggVec() { return null; }

    @Override
    public void setAggVec(Map aggVec) { }
}
