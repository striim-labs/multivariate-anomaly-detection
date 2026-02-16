"""
Kafka Producer for SMD Telemetry Streaming

Reads a machine's test data row by row and publishes JSON messages
to the telemetry_stream Kafka topic.

"""

import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("SMD Telemetry Producer - not yet implemented")
    logger.info(f"MACHINE_ID: {os.getenv('MACHINE_ID', 'machine-1-1')}")
    raise NotImplementedError("Producer not yet implemented. See Phase 5.")


if __name__ == "__main__":
    main()
