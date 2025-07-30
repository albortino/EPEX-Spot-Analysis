# Strompreis Analyse

This project is designed to analyze electricity prices, providing insights and tools for understanding energy costs.
## Project Overview

This project provides a comprehensive analysis of electricity prices, focusing on the Austrian and German market and offering tools for understanding and optimizing energy costs. It includes functionalities for backtesting electricity tariffs, particularly those offered by dynamic pricing providers like aWATTar.

## EPEX Spot Backtesting

One of the core features of this project is the ability to backtest electricity tariffs, especially those with dynamic pricing components. This allows users to simulate their historical electricity costs under different tariff structures.

To utilize this feature:

1.  **Upload Consumption Data:** You will need to upload your historical electricity consumption data. This data is typically available from your energy network provider. If you have questions about the required format or where to obtain this data, please refer to the `https://awattar-backtesting.github.io` documentation.

2.  **Configure Tariffs:**
    *   **Pre-selected Configuration:** The application provides pre-selected tariff configurations on the left sidebar. These configurations are based on an analysis of common tariffs (e.g., those found on Ecotricity) as of mid-July 2025. Please be aware that these tariffs are subject to change.
    *   **Custom Configuration:** For individual tariffs or specific scenarios, you can adapt the values in the configuration section to match your unique tariff structure.

## Tab Views

The application is organized into five distinct tab views, each serving a specific purpose:

*   **Spot Price Analysis:** This tab focuses on analyzing historical electricity spot prices, providing insights into price trends and volatility.
*   **Cost Comparison:** Here, you can compare the costs of different electricity tariffs based on your consumption data.
*   **Usage Pattern Analysis:** This section visualizes your electricity consumption patterns, helping you understand when and how you use energy.
*   **Yearly Summary:** Provides an annual overview of your electricity costs and consumption.
*   **Download Data:** Allows you to download processed data and analysis results.
## About the Author

This project is developed by **Alborino** (GitHub: [alborino](https://github.com/alborino)).

## Environment Installation

To ensure reproducibility and ease of setup, this project uses a Conda environment.

### Getting the Conda Environment

To set up the project's environment on your machine, you can use the `environment.yml` file.

1.  **Create the environment:**

    ```bash
    conda env create -f environment.yml
    ```

    This command will create a new Conda environment with the name specified in the `environment.yml` file (usually `strompreis-analyse` or similar).

2.  **Activate the environment:**

    ```bash
    conda activate strompreis-analyse # Replace 'strompreis-analyse' with your environment name if different
    ```

    Once activated, you can run the project's scripts and notebooks within this isolated environment.
