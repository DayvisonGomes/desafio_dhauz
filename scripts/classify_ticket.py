"""CLI: classify a single ticket (placeholder to wire components)"""
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ticket", type=str, help="Ticket text to classify")
    parser.add_argument("--mode", choices=["rag","hybrid"], default="hybrid")
    args = parser.parse_args()

    # This script is a light wrapper; user must ensure models/vectorstore are available
    print("This script is a placeholder. Use package to load models and classify programmatically.")
    print(f"Ticket: {args.ticket}")

if __name__ == '__main__':
    main()
