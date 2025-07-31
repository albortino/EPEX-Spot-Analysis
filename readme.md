
# Electricity Tariff Comparison Dashboard

This Streamlit application provides a comprehensive tool for analyzing your electricity consumption, comparing different tariff options (flexible vs. static), simulating the impact of peak load shifting, and gaining insights into your energy usage patterns.

## Description

The dashboard allows users to upload their historical electricity consumption data (typically in CSV format) and compare the cost-effectiveness of various electricity tariffs. It fetches real-time EPEX spot prices for a selected country and overlays them with your consumption data. Key features include identifying the cheapest available tariffs, simulating savings through peak load shifting, and visualizing detailed consumption breakdowns and price trends.

## Key Features

### Consumption Data Upload

Easily upload your historical electricity consumption data via a CSV file. The application supports parsing common formats for providers and can be extended for others.

### **Dynamic Tariff Comparison**

* **Automatic Cheapest Tariff Detection:** Automatically identifies the most economical predefined flexible and static tariffs based on your uploaded consumption data.
* **Custom Tariff Configuration:** Allows users to define their own flexible and static tariffs by specifying on-top prices, variable percentages, and monthly fees.

### **Peak Load Shifting Stimulation**

Simulate the potential cost savings achievable by shifting a configurable percentage of your peak electricity consumption to off-peak hours within a flexible time window.

### **Interactive Analysis Tabs**

1) **Spot Price Analysis:**
    * Visualize the distribution of EPEX spot prices over time (hourly, weekly, or monthly).
    * Analyze average spot prices through an interactive heatmap showing daily and seasonal patterns.
2) **Cost Comparison:**
    * Compare total electricity costs and the effective price per kWh for different tariffs over daily, weekly, or monthly periods.
    * Review a detailed table summarizing consumption, costs, and savings.
3) **Usage Pattern Analysis:**
    * Classify your hourly consumption into `Base Load`, `Regular Load`, and `Peak Load` based on dynamic thresholds.
    * Understand the contribution of each load type to your total consumption and their associated average prices.
    * Visualize consumption patterns with a daily breakdown for a selected example day.
4) **Yearly Summary:**
    * Provides an aggregated overview of total consumption and costs for each year present in your data.
5) **Download Data:**
    * Export the detailed analysis results and raw EPEX spot price data in XLSX format for further offline analysis.

### **Intelligent Recommendations**

Receive data-driven recommendations on which tariff type (flexible or static) is likely to be more cost-effective for your usage profile, along with optimization tips.

### **Absence Day Handling**

Option to exclude days with unusually low consumption, improving the accuracy of baseline analysis and recommendations.

## How to Use

1. **Clone the Repository**

```bash
git clone <repository_url>
cd your-repository-directory
```

2. **Install Dependencies**

Ensure you have Python and Conda (Anaconda or Miniconda) installed. Then, install the required libraries:

```bash
conda env create -f environment.yml
```

3. **Run the Application**

Launch the Streamlit application from your terminal:

```bash
streamlit run app.py
```

4. **Upload Consumption Data**

In the sidebar of the application, use the "Upload Your Consumption CSV" button to upload your electricity usage data. For best results, ensure your data is in hourly or 15-minute intervals.

5. **Configure Settings**

* **Country:** Select the country for which EPEX spot prices should be fetched (e.g., Germany, Austria).
* **Analysis Period:** Define the start and end dates for your analysis.
* **Tariff Plans:** Choose to automatically compare the cheapest predefined tariffs or manually configure your own flexible and static tariff details.
* **Peak Load Shifting:** Adjust the slider to simulate shifting a percentage of your peak consumption.

6. **Explore Insights**

Navigate through the different tabs ("Spot Price Analysis", "Cost Comparison", "Usage Pattern Analysis", "Yearly Summary", "Download Data") to view detailed charts, tables, and recommendations.

## Data Requirements

* **Consumption Data:** A CSV file containing your electricity consumption. Key columns expected are a timestamp and a consumption value (e.g., `consumption_kwh`). The application is designed to be flexible with data granularity, but 15-minute intervals provide richer insights for usage pattern analysis. In case of any uncertainties please refer to the documentation in [awattar backtesting](https://awattar-backtesting.github.io/).
* **Spot Prices:** The application automatically fetches hourly EPEX spot prices from the aWATTar API for the selected country and date range. These prices are cached locally to optimize performance.

## File Structure Overview

* `app.py`: The main script orchestrating the Streamlit application flow, loading modules, and rendering the UI.
* `methods/`: Contains all supporting Python modules.
    * `config.py`: Stores application-wide constants, configurations, and API-related settings.
    * `data_loader.py`: Manages fetching EPEX spot prices from the aWATTar API and processing user-uploaded consumption data. Includes caching mechanisms.
    * `analysis.py`: Implements the core analytical logic, including consumption classification (base, peak, regular) and peak load shifting simulation.
    * `ui_components.py`: Handles the creation and rendering of all user interface elements, including sidebars, tabs, charts, and user inputs.
    * `tariffs.py`: Defines the `Tariff` class and `TariffManager` for managing and calculating costs associated with different tariff structures.
    * `utils.py`: Contains general utility functions used across the application (e.g., date handling, file export).
    * `file_parser.py`: Houses the `ConsumptionDataParser` class responsible for parsing various CSV formats of consumption data. Interacts with [awattar backtesting](https://awattar-backtesting.github.io/) to use the exisitng javascript parsing logic.
    * `charts.py`: Provides custom functions for generating specific plot types used in the dashboard.

## Acknowledgements

This project was influenced by and extends the functionality presented in the [awattar backtesting](https://awattar-backtesting.github.io/) project, providing a more detailed and user-friendly interface for electricity tariff analysis.
