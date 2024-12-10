# Imports
import random
import time
from dataclasses import dataclass
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from faker import Faker
from scipy import stats
from scipy.linalg import LinAlgError, cholesky
from scipy.stats import norm

# Type aliases
ParamDict = Dict[str, Any]
CorrelationDict = Dict[str, Union[str, float]]


class FieldType(Enum):
    """Enumeration for field types."""

    CONTINUOUS = "Continuous"
    CATEGORICAL = "Categorical"


class DataType(Enum):
    """Enumeration for data types."""

    (
        INTEGER,
        FLOAT,
        PERCENTAGE,
        CURRENCY,
        TIME_SERIES,
    ) = (
        "Integer",
        "Float",
        "Percentage",
        "Currency",
        "Time Series",
    )
    NORMAL_DIST, LOG_SCALE, EXP_DIST, LAT_LONG = (
        "Normal Distribution",
        "Logarithmic Scale",
        "Exponential Distribution",
        "Latitude and Longitude",
    )
    BOOLEAN, GENDER, COUNTRY, JOB_TITLE, INDUSTRY = (
        "Boolean",
        "Gender",
        "Country",
        "Job Title",
        "Industry",
    )
    CUSTOM_LIST, ORDINAL, EMAIL, COMPANY, POSTAL = (
        "Custom List",
        "Ordinal",
        "Email Domains",
        "Company Names",
        "Postal Codes",
    )


@dataclass
class ColumnConfig:
    """Configuration for a dataset column."""

    name: str
    type: DataType
    params: ParamDict
    correlations: List[CorrelationDict]


class DataGenerator:
    """Generates synthetic data based on column configurations."""

    CORRELATION_TYPES = {
        "Positive High": (0.7, 1.0),
        "Positive Average": (0.4, 0.69),
        "Positive Low": (0.1, 0.39),
        "Negative High": (-1.0, -0.7),
        "Negative Average": (-0.69, -0.4),
        "Negative Low": (-0.39, -0.1),
    }

    def __init__(self):
        """Initialize with Faker instance and data type generators."""
        self.fake = Faker()
        self._setup_generators()

    def _setup_generators(self):
        """Map data types to their generator functions."""
        self.generators = {
            DataType.INTEGER: self._generate_integer,
            DataType.FLOAT: self._generate_float,
            DataType.PERCENTAGE: self._generate_percentage,
            DataType.CURRENCY: self._generate_currency,
            DataType.TIME_SERIES: self._generate_time_series,
            DataType.NORMAL_DIST: self._generate_normal,
            DataType.LOG_SCALE: self._generate_logarithmic,
            DataType.EXP_DIST: self._generate_exponential,
            DataType.LAT_LONG: self._generate_lat_long,
            DataType.COUNTRY: self._generate_country,
            DataType.JOB_TITLE: self._generate_job_title,
            DataType.INDUSTRY: self._generate_industry,
            DataType.EMAIL: self._generate_email,
            DataType.POSTAL: self._generate_postal,
            DataType.COMPANY: self._generate_company,
            DataType.GENDER: self._generate_gender,
            DataType.BOOLEAN: self._generate_boolean,
            DataType.CUSTOM_LIST: self._generate_custom_list,
            DataType.ORDINAL: self._generate_ordinal,
        }

    def _generate_integer(self, params: ParamDict, num_rows: int) -> pd.Series:
        min_val, max_val = params.get("Min", 0), params.get("Max", 100)
        mean = params.get("Mean", (min_val + max_val) / 2)
        std = params.get("Standard Deviation", (max_val - min_val) / 6)
        print(
            f"DEBUG: Min={min_val}, Max={max_val}, Mean={mean}, Std={std}"
        )  # Debug log
        if min_val >= max_val:
            raise ValueError("Min must be less than Max.")
        use_normal = params.get("Use Normal Distribution", False)
        if use_normal:
            values = np.random.normal(mean, std, num_rows)
            clipped_values = np.clip(values, min_val, max_val)
            return pd.Series(clipped_values.round().astype(int))
        else:
            values = np.random.randint(min_val, max_val, num_rows)
            clipped_values = np.clip(values, min_val, max_val)
            return pd.Series(clipped_values)

    def _generate_float(self, params: ParamDict, num_rows: int) -> pd.Series:
        min_val, max_val, decimals = (
            params.get("Min", 0.0),
            params.get("Max", 100.0),
            params.get("Decimals", 2),
        )
        mean = params.get("Mean", (min_val + max_val) / 2)
        std = params.get("Standard Deviation", (max_val - min_val) / 6)
        print(
            f"DEBUG: Min={min_val}, Max={max_val}, Mean={mean}, Std={std}"
        )  # Debug log
        if min_val >= max_val:
            raise ValueError("Min must be less than Max.")
        use_normal = params.get("Use Normal Distribution", False)
        if use_normal:
            values = np.random.normal(mean, std, num_rows)
            clipped_values = np.clip(values, min_val, max_val)
        else:
            values = np.random.uniform(min_val, max_val, num_rows)
            clipped_values = np.clip(values, min_val, max_val)
        return pd.Series(np.round(clipped_values, decimals))

    def _generate_percentage(self, params: ParamDict, num_rows: int) -> pd.Series:
        min_val, max_val, decimals = (
            max(0, params.get("Min", 0)),
            min(100, params.get("Max", 100)),
            params.get("Decimals", 2),
        )
        return pd.Series(
            np.round(np.random.uniform(min_val, max_val, num_rows), decimals)
        )

    def _generate_currency(self, params: ParamDict, num_rows: int) -> pd.Series:
        min_val, max_val, decimals, currency_symbol = (
            params.get("Min", 0.0),
            params.get("Max", 1000.0),
            params.get("Decimals", 2),
            params.get("Currency Symbol", "$"),
        )
        values = np.round(np.random.uniform(min_val, max_val, num_rows), decimals)
        return pd.Series([f"{currency_symbol}{value:,.2f}" for value in values])

    def _generate_time_series(self, params: ParamDict, num_rows: int) -> pd.Series:
        start_date, frequency, trend = (
            pd.to_datetime(params.get("Start Date", "2024-01-01")),
            params.get("Frequency", "D"),
            params.get("Trend", "linear"),
        )
        seasonality, noise_level = (
            params.get("Seasonality", None),
            params.get("Noise", 0.1),
        )
        dates = pd.date_range(start=start_date, periods=num_rows, freq=frequency)
        values = self._apply_time_series_trend(trend, num_rows)
        if seasonality:
            values += self._apply_seasonality(seasonality, num_rows)
        return pd.Series(
            values + np.random.normal(0, noise_level * np.std(values), num_rows),
            index=dates,
        )

    def _apply_time_series_trend(self, trend: str, num_rows: int) -> np.ndarray:
        if trend == "linear":
            return np.linspace(0, 100, num_rows)
        return np.exp(np.linspace(0, 4, num_rows))

    def _apply_seasonality(self, seasonality: str, num_rows: int) -> np.ndarray:
        period = {"daily": 24, "weekly": 7, "monthly": 30, "yearly": 12}[seasonality]
        return np.sin(np.linspace(0, 2 * np.pi * num_rows / period, num_rows)) * 20

    def _generate_normal(self, params: ParamDict, num_rows: int) -> pd.Series:
        mean, std, decimals = (
            params.get("Mean", 0.0),
            params.get("Standard Deviation", 1.0),
            params.get("Decimals", 2),
        )
        return pd.Series(np.round(np.random.normal(mean, std, num_rows), decimals))

    def _generate_logarithmic(self, params: ParamDict, num_rows: int) -> pd.Series:
        scale, decimals = params.get("Scale", 1.0), params.get("Decimals", 2)
        return pd.Series(
            np.round(np.random.lognormal(0, 1, num_rows) * scale, decimals)
        )

    def _generate_exponential(self, params: ParamDict, num_rows: int) -> pd.Series:
        scale, decimals = params.get("Scale", 1.0), params.get("Decimals", 2)
        return pd.Series(np.round(np.random.exponential(scale, num_rows), decimals))

    def _generate_lat_long(self, params: ParamDict, num_rows: int) -> pd.Series:
        lat_min, lat_max, long_min, long_max = (
            params.get("Lat Min", -90),
            params.get("Lat Max", 90),
            params.get("Long Min", -180),
            params.get("Long Max", 180),
        )
        lats, longs = (
            np.random.uniform(lat_min, lat_max, num_rows),
            np.random.uniform(long_min, long_max, num_rows),
        )
        return pd.Series([f"({lat:.6f}, {long:.6f})" for lat, long in zip(lats, longs)])

    def _generate_country(self, params: ParamDict, num_rows: int) -> pd.Series:
        return pd.Series([self.fake.country() for _ in range(num_rows)])

    def _generate_job_title(self, params: ParamDict, num_rows: int) -> pd.Series:
        return pd.Series([self.fake.job() for _ in range(num_rows)])

    def _generate_industry(self, params: ParamDict, num_rows: int) -> pd.Series:
        industries = [
            "Technology",
            "Healthcare",
            "Finance",
            "Education",
            "Manufacturing",
            "Retail",
            "Entertainment",
            "Energy",
            "Transportation",
            "Agriculture",
        ]
        return pd.Series(np.random.choice(industries, num_rows))

    def _generate_email(self, params: ParamDict, num_rows: int) -> pd.Series:
        return pd.Series([self.fake.email() for _ in range(num_rows)])

    def _generate_postal(self, params: ParamDict, num_rows: int) -> pd.Series:
        return pd.Series([self.fake.zipcode() for _ in range(num_rows)])

    def _generate_company(self, params: ParamDict, num_rows: int) -> pd.Series:
        return pd.Series([self.fake.company() for _ in range(num_rows)])

    def _generate_gender(self, params: ParamDict, num_rows: int) -> pd.Series:
        return pd.Series(
            [
                self.fake.random_element(elements=("Male", "Female", "Other"))
                for _ in range(num_rows)
            ]
        )

    def _generate_boolean(self, params: ParamDict, num_rows: int) -> pd.Series:
        return pd.Series(np.random.choice([True, False], num_rows))

    def _generate_custom_list(self, params: dict, num_rows: int) -> pd.Series:
        """
        Generate a custom list based on user-defined values and their corresponding percentages,
        and debug the result for percentages, means, and standard deviations.
        Args:
            params (dict): A dictionary containing "Value Names", "Percentages",
                        "Individual Means", and "Individual Standard Deviations".
            num_rows (int): Number of rows in the generated dataset.
        Returns:
            pd.Series: A Pandas Series with values generated based on user-defined rules.
        """
        values = params["Value Names"]
        percentages = params["Percentages"]
        individual_means = params["Individual Means"]
        individual_stds = params["Individual Standard Deviations"]
        # Validate inputs
        if not np.isclose(sum(percentages), 100.0):
            raise ValueError("Percentages must sum to 100.")
        if not (
            len(values)
            == len(percentages)
            == len(individual_means)
            == len(individual_stds)
        ):
            raise ValueError(
                "The number of values, percentages, means, and standard deviations must match."
            )
        if num_rows <= 0:
            raise ValueError("The number of rows must be greater than zero.")
        if not all(isinstance(m, (int, float)) for m in individual_means):
            raise ValueError("All means must be numeric.")
        if not all(isinstance(s, (int, float)) for s in individual_stds):
            raise ValueError("All standard deviations must be numeric.")
        # Convert percentages to exact counts, with rounding adjustments
        expected_counts = (np.array(percentages) / 100 * num_rows).astype(int)
        remainder = num_rows - sum(expected_counts)
        # Handle the remainder by distributing it to the categories proportionally
        if remainder > 0:
            # Find the largest discrepancies (rounded down counts) and add remainder there
            sorted_indices = np.argsort(percentages)
            for i in sorted_indices[-remainder:]:
                expected_counts[i] += 1
        # Generate values
        generated_values = []
        for value_idx, count in enumerate(expected_counts):
            # Generate Gaussian values for the given count
            if isinstance(values[value_idx], (int, float)):
                values_for_idx = np.random.normal(
                    loc=individual_means[value_idx],
                    scale=individual_stds[value_idx],
                    size=count,
                )
            else:
                values_for_idx = [values[value_idx]] * count  # Non-numeric values
            generated_values.extend(values_for_idx)
        # Shuffle generated values to avoid order bias
        np.random.shuffle(generated_values)
        result_series = pd.Series(generated_values)
        # Debugging: Analyze the result
        debug_info = []
        total_count = len(result_series)
        for value in values:
            value_rows = result_series[result_series == value]
            count = len(value_rows)
            percentage = (count / total_count) * 100
            # Only calculate mean and standard deviation for numeric data
            if isinstance(value, (int, float)):
                mean = value_rows.mean() if len(value_rows) > 0 else None
                std = value_rows.std(ddof=0) if len(value_rows) > 1 else None
            else:
                mean = "N/A"
                std = "N/A"
            debug_info.append(
                {
                    "Value Name": value,
                    "Count": count,
                    "Percentage": round(percentage, 2),
                    "Mean": mean if mean is not None else "N/A",
                    "Standard Deviation": std if std is not None else "N/A",
                    "Expected Mean": individual_means[values.index(value)],
                    "Expected Std Dev": individual_stds[values.index(value)],
                }
            )
        # Convert the debug info to a pandas DataFrame for better display
        debug_df = pd.DataFrame(debug_info)
        # Print debug information in tabular format
        st.table(debug_df)
        return result_series

    def _generate_ordinal(self, params: ParamDict, num_rows: int) -> pd.Series:
        categories = params.get("Categories", "Low,Medium,High").split(",")
        return pd.Series(np.random.choice(categories, size=num_rows))

    def validate_correlation_matrix(corr_matrix):
        """Validate if the correlation matrix is positive semidefinite."""
        try:
            # Try Cholesky decomposition to ensure positive semidefiniteness
            cholesky(corr_matrix)
            return True
        except LinAlgError:
            return False

    def generate_correlated_data(
        self, base_series: pd.Series, correlation: float, num_rows: int
    ) -> pd.Series:
        """Efficiently generate correlated data."""
        base_mean, base_std = base_series.mean(), base_series.std()
        base_standardized = (base_series - base_mean) / base_std
        random_data = np.random.normal(0, 1, num_rows)
        # Vectorized correlation adjustment
        correlated = (
            correlation * base_standardized + np.sqrt(1 - correlation**2) * random_data
        )
        result = correlated * base_std + base_mean
        return pd.Series(result, dtype=np.float32)

    def generate_data(
        self,
        column_config: ColumnConfig,
        num_rows: int,
        existing_data: Optional[Dict[str, pd.Series]] = None,
    ) -> pd.Series:
        """Optimized data generation with optional correlation handling."""
        params = column_config.params
        print(f"DEBUG: column_config.params = {params}")
        if column_config.type == DataType.INTEGER:
            return self._generate_integer(params, num_rows)
        generator = self.generators.get(column_config.type)
        if not generator:
            raise ValueError(
                f"No generator defined for data type: {column_config.type}"
            )
        # Generate base data using vectorized method
        base_data = generator(column_config.params, num_rows)
        # Quick correlation application
        if existing_data is not None and column_config.correlations:
            base_data = self.apply_correlations(
                base_data, column_config.correlations, existing_data, num_rows
            )
        return base_data

    def apply_correlations(self, base_data, correlations, existing_data, num_rows):
        """More efficient correlation application with Cholesky decomposition."""
        if not correlations:
            return base_data
        # Extract relevant columns and create standardized matrix
        standardized_columns = []
        for corr in correlations:
            base_column = corr["Base Column"]
            if base_column in existing_data:
                # Standardize the column if it exists in the existing_data
                standardized_columns.append(
                    (existing_data[base_column] - existing_data[base_column].mean())
                    / existing_data[base_column].std()
                )
            else:
                # Log or handle the case where the column doesn't exist in existing_data
                print(f"Warning: Column '{base_column}' not found in existing_data.")
                # Handle missing column (e.g., skip the correlation or use default behavior)
                standardized_columns.append(
                    np.zeros(num_rows)
                )  # Default to zero if column is missing
        base_standardized = (base_data - base_data.mean()) / base_data.std()
        standardized_matrix = np.column_stack(
            [base_standardized, *standardized_columns]
        )
        print("Exisiting Data Columns:", existing_data.keys())
        # Construct correlation matrix
        correlation_matrix = np.eye(len(correlations) + 1)
        for i, corr in enumerate(correlations):
            # Get the correlation range from CORRELATION_TYPES
            correlation_range = self.CORRELATION_TYPES.get(
                corr.get("Correlation Value", "No Correlation"), (0, 0)
            )
            # Randomly select a single correlation value for each correlation type
            corr_value = random.uniform(correlation_range[0], correlation_range[1])
            # Apply the same correlation value to the matrix for both directions (symmetric)
            correlation_matrix[0, i + 1] = corr_value
            correlation_matrix[i + 1, 0] = corr_value

        def nearest_psd(matrix):
            eigvals, eigvecs = np.linalg.eigh(matrix)
            eigvals[eigvals < 0] = 0
            return eigvecs @ np.diag(eigvals) @ eigvecs.T

        correlation_matrix = nearest_psd(correlation_matrix)
        try:
            cholesky_matrix = cholesky(correlation_matrix)
        except LinAlgError:
            raise ValueError("Correlation matrix is not positive semi-definite")
        # Generate correlated data using Cholesky matrix
        correlated_matrix = standardized_matrix @ cholesky_matrix.T
        result = (correlated_matrix[:, 0] * base_data.std()) + base_data.mean()
        return pd.Series(result, dtype=np.float32)


class DatasetUI:
    """
    Handles the user interface for dataset configuration and generation.
    """

    CORRELATION_TYPES = {
        "Positive High": (0.7, 1.0),
        "Positive Average": (0.4, 0.69),
        "Positive Low": (0.1, 0.39),
        "Negative High": (-1.0, -0.7),
        "Negative Average": (-0.69, -0.4),
        "Negative Low": (-0.39, -0.1),
    }

    def __init__(self):
        """Initialize the UI components and session state."""
        self.data_generator = DataGenerator()
        self._initialize_session_state()

    def _initialize_session_state(self):
        """Initialize or reset the session state variables."""
        defaults = {
            "num_rows": 5,
            "num_columns": 3,
            "columns": [],
            "df": pd.DataFrame(),
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def _create_number_input(self, label, min_value, max_value, value, step, key):
        """Helper function to create a number input widget."""
        return st.number_input(
            label,
            min_value=float(min_value),
            max_value=float(max_value),
            value=float(value),
            step=float(step),
            key=key,
        )

    def _create_checkbox(self, label, key, help_text=""):
        """Helper function to create a checkbox widget."""
        return st.checkbox(label, key=key, help=help_text)

    def sort_columns_by_dependencies(self, columns):
        """
        Sort columns to ensure dependencies are resolved.
        Uses a topological sort-like approach.
        """
        sorted_columns = []
        remaining_columns = columns[:]
        resolved_columns = set()
        while remaining_columns:
            for column in remaining_columns:
                if not column.correlations or all(
                    corr["Base Column"] in resolved_columns
                    for corr in column.correlations
                ):
                    sorted_columns.append(column)
                    resolved_columns.add(column.name)
                    remaining_columns.remove(column)
                    break
        return sorted_columns

    def _render_type_specific_params(self, idx, data_type):
        """Render parameter inputs based on the data type."""
        params = {}
        # Always show min and max for numeric types
        if data_type in [
            DataType.INTEGER,
            DataType.FLOAT,
            DataType.PERCENTAGE,
            DataType.CURRENCY,
        ]:
            if data_type in [DataType.INTEGER, DataType.FLOAT]:
                col3, col4 = st.columns(2)
                with col3:
                    # Minimum value input
                    params["Min"] = self._create_number_input(
                        "Minimum", -1000000.0, 1000000.0, 0.0, 0.1, f"min_{idx}"
                    )
                with col4:
                    # Maximum value input
                    params["Max"] = self._create_number_input(
                        "Maximum", -1000000.0, 1000000.0, 100.0, 0.1, f"max_{idx}"
                    )
                # For both Integer and Float types, show mean and standard deviation inputs
                col5, col6 = st.columns(2)
                with col5:
                    params["Mean"] = self._create_number_input(
                        "Mean",
                        -1000000.0,
                        1000000.0,
                        (params.get("Min", 0) + params.get("Max", 100)) / 2,
                        0.1,
                        f"mean_{idx}",
                    )
                with col6:
                    params["Standard Deviation"] = self._create_number_input(
                        "Standard Deviation",
                        0.1,
                        1000000.0,
                        (params.get("Max", 100) - params.get("Min", 0)) / 6,
                        0.1,
                        f"std_{idx}",
                    )
                # Option to use normal distribution for Integer and Float types
                use_normal = self._create_checkbox(
                    "Use Normal Distribution",
                    key=f"use_normal_{idx}",
                    help_text="Generate data following a normal distribution",
                )
                params["Use Normal Distribution"] = use_normal
        elif data_type == DataType.PERCENTAGE:
            col1, col2 = st.columns(2)
            with col1:
                params["Min"] = self._create_number_input(
                    "Minimum (%)", 0.0, 100.0, 0.0, 0.1, f"min_{idx}"
                )
            with col2:
                params["Max"] = self._create_number_input(
                    "Maximum (%)", 0.0, 100.0, 100.0, 0.1, f"max_{idx}"
                )
        elif data_type == DataType.CURRENCY:
            col1, col2 = st.columns(2)
            with col1:
                params["Min"] = self._create_number_input(
                    "Minimum", 0.0, 1000000.0, 0.0, 0.01, f"min_{idx}"
                )
            with col2:
                params["Max"] = self._create_number_input(
                    "Maximum", 0.0, 1000000.0, 1000.0, 0.01, f"max_{idx}"
                )
            params["Currency Symbol"] = st.text_input(
                "Currency Symbol", value="$", key=f"symbol_{idx}"
            )
        elif data_type == DataType.TIME_SERIES:
            col1, col2 = st.columns(2)
            with col1:
                params["Start Date"] = st.date_input(
                    "Start Date", value=date.today(), key=f"start_date_{idx}"
                )
            with col2:
                params["Frequency"] = st.selectbox(
                    "Frequency", options=["D", "M", "Y"], key=f"freq_{idx}"
                )
        elif data_type == DataType.CUSTOM_LIST:
            st.subheader("Configure Custom List")
            # Number of values
            num_values = st.number_input(
                "Number of Values",
                min_value=1,
                max_value=st.session_state.num_rows,
                value=3,
                step=1,
                key=f"num_values_{idx}",
            )
            params["Num Values"] = num_values
            # Initialize lists for Value Names, Percentages, Means, and Standard Deviations
            value_names = []
            percentages = []
            individual_means = []
            individual_stds = []
            # Dynamic inputs for each value
            for i in range(num_values):
                st.write(f"### Configuration for Value {i + 1}")
                # Name of the value
                value_name = st.text_input(
                    f"Name for Value {i + 1}",
                    value=f"Value_{i + 1}",
                    key=f"value_name_{idx}_{i}",
                )
                value_names.append(value_name)
                # Percentage for the value
                percentage = st.number_input(
                    f"Percentage for {value_name} (leave blank for auto-calculation)",
                    min_value=0.0,
                    max_value=100.0,
                    step=1.0,
                    key=f"percentage_{idx}_{i}",
                    format="%.2f",
                )
                percentages.append(percentage if percentage > 0 else None)
                # Mean for the value
                mean = st.number_input(
                    f"Mean for {value_name}",
                    value=0.0,
                    key=f"mean_{idx}_{i}",
                )
                individual_means.append(mean)
                # Standard Deviation for the value
                std = st.number_input(
                    f"Standard Deviation for {value_name}",
                    value=1.0,
                    min_value=0.001,
                    key=f"std_{idx}_{i}",
                )
                individual_stds.append(std)
            # Auto-fill remaining percentage
            remaining_percentage = 100.0 - sum(p for p in percentages if p is not None)
            unassigned_count = percentages.count(None)
            if unassigned_count > 0:
                auto_fill_value = remaining_percentage / unassigned_count
                percentages = [
                    p if p is not None else auto_fill_value for p in percentages
                ]
            # Validate percentages
            total_percentage = sum(percentages)
            if not np.isclose(total_percentage, 100.0):
                st.warning(
                    f"Total percentage must equal 100%. Currently: {total_percentage:.2f}%"
                )
            else:
                st.success(f"Total percentage is valid: {total_percentage:.2f}%")
            # Display remaining percentage
            st.markdown(
                f"**Remaining percentage auto-filled: {remaining_percentage:.2f}%**"
            )
            # Store parameters
            params["Value Names"] = value_names
            params["Percentages"] = percentages
            params["Individual Means"] = individual_means
            params["Individual Standard Deviations"] = individual_stds
        st.session_state.columns[idx].params = params

    def _render_column_management(self):
        """Render column management controls."""
        st.write("## Column Management")
        if st.button("Add New Column", key="add_column_btn"):
            new_idx = len(st.session_state.columns)
            st.session_state.columns.append(
                ColumnConfig(
                    name=f"Column_{new_idx + 1}",
                    type=DataType.INTEGER,
                    params={"Min": 0, "Max": 100},
                    correlations=[],
                )
            )
            st.session_state.num_columns += 1
        if st.session_state.columns:
            cols_to_delete = st.multiselect(
                "Select columns to delete",
                options=[col.name for col in st.session_state.columns],
                key="delete_cols",
            )
            if cols_to_delete:
                st.session_state.columns = [
                    col
                    for col in st.session_state.columns
                    if col.name not in cols_to_delete
                ]
                st.session_state.num_columns = len(st.session_state.columns)

    def render_configuration_ui(self):
        """Render the main configuration interface."""
        st.title("Dr.Shah's Dataset Generator")
        with st.sidebar:
            st.title("Help Information")
            st.markdown("""
                ### How to use this app
                1. Set the number of rows and columns.
                2. Configure each column's data type and parameters.
                3. Set correlations between numeric columns.
                4. Generate and analyze your dataset.
            """)
        self._render_row_column_config()
        self._render_column_management()
        if st.session_state.columns:
            self._render_column_config()
            # 🔑 Add this line to show correlation configuration
            self._render_correlation_config()
            self._render_generation_section()

    def _render_row_column_config(self):
        """Render the row and column configuration section."""
        st.write("## Step 1: Set Number of Rows and Columns")
        col1, col2 = st.columns(2)
        with col1:
            num_rows = st.number_input(
                "Number of Rows",
                min_value=1,
                max_value=1_000_000,
                value=st.session_state.num_rows,
                step=1,
                help="Specify the number of rows in the dataset.",
                key="num_rows_input",  # Add a unique key
            )
        with col2:
            num_columns = st.number_input(
                "Number of Columns",
                min_value=1,
                max_value=50,
                value=st.session_state.num_columns,
                step=1,
                help="Specify the number of columns to include in the dataset.",
                key="num_columns_input",  # Add a unique key
            )
        # Use a flag in session state instead of rerun
        if st.button("Proceed to Column Configuration", key="proceed_columns"):
            st.session_state.proceed_to_columns = True
            st.session_state.num_rows = num_rows
            st.session_state.num_columns = num_columns
            self._initialize_columns(num_rows, num_columns)

    def _initialize_columns(self, num_rows, num_columns):
        """Initialize column configurations based on dimensions."""
        st.session_state.num_rows = num_rows
        st.session_state.num_columns = num_columns
        if not st.session_state.columns:
            st.session_state.columns = [
                ColumnConfig(
                    name=f"Column_{i + 1}",
                    type=DataType.INTEGER,
                    params={"Min": 0, "Max": 100},
                    correlations=[],
                )
                for i in range(num_columns)
            ]

    def _render_single_column_config(self, idx, column):
        """Render configuration options for a single column."""
        col1, col2 = st.columns(2)
        with col1:
            # Column name input
            new_name = st.text_input(
                "Column Name", value=column.name, key=f"col_name_{idx}"
            )
            st.session_state.columns[idx].name = new_name
            # Field type selection (Continuous/Categorical)
            field_type = st.selectbox(
                "Field Type",
                options=[t.value for t in FieldType],
                index=0 if column.type in [t.value for t in FieldType] else 1,
                key=f"field_type_{idx}",
            )
        with col2:
            # Data type selection based on field type
            available_types = [
                t.value
                for t in DataType
                if (
                    field_type == FieldType.CONTINUOUS.value
                    and t.value
                    in [
                        "Integer",
                        "Float",
                        "Percentage",
                        "Currency",
                        "Time Series",
                    ]
                )
                or (
                    field_type == FieldType.CATEGORICAL.value
                    and t.value
                    in ["Boolean", "Gender", "Country", "Job Title", "Custom List"]
                )
            ]
            data_type = st.selectbox(
                "Data Type",
                options=available_types,
                index=available_types.index(column.type)
                if column.type in available_types
                else 0,
                key=f"data_type_{idx}",
            )
            st.session_state.columns[idx].type = DataType(data_type)
        # Render parameters based on data type
        self._render_type_specific_params(idx, DataType(data_type))

    def _render_column_config(self):
        """Render the column configuration section."""
        st.write("## Step 2: Configure Columns")
        for idx, column in enumerate(st.session_state.columns):
            with st.expander(f"Column {idx + 1}: {column.name}", expanded=True):
                self._render_single_column_config(idx, column)

    def _render_correlation_config(self): 
        st.write("## Step 3: Configure Correlations")
        # Filter for numeric columns
        numeric_columns = [
            col
            for col in st.session_state.columns
            if col.type in [
                DataType.INTEGER,
                DataType.FLOAT,
                DataType.PERCENTAGE,
            ]
        ]
        if len(numeric_columns) < 2:
            st.info("Add at least two numeric columns to configure correlations.")
            return

        # Use the provided CORRELATION_TYPES dictionary
        CORRELATION_TYPES = {
            "Positive High": (0.7, 1.0),
            "Positive Average": (0.4, 0.69),
            "Positive Low": (0.1, 0.39),
            "Negative High": (-1.0, -0.7),
            "Negative Average": (-0.69, -0.4),
            "Negative Low": (-0.39, -0.1),
        }

        with st.container(border=True):
            st.write("### Correlation Matrix Configuration")
            st.markdown("""
            - Select correlation types between numeric columns
            - 🟢 **Positive Correlation**: Variables move together
            - 🔴 **Negative Correlation**: Variables move in opposite directions
            - 🔷 **No Correlation**: Variables are independent
            """)

            # Initialize correlation matrix with actual numeric values
            correlation_matrix = np.full(
                (len(numeric_columns), len(numeric_columns)), None
            )
            correlation_value_matrix = np.full(
                (len(numeric_columns), len(numeric_columns)), "No Correlation"
            )
            correlation_options = ["No Correlation"] + list(CORRELATION_TYPES.keys())
            
            # Create a grid of selectboxes for each pair of columns
            for row in range(len(numeric_columns)):
                cols = st.columns(len(numeric_columns))
                for col in range(len(numeric_columns)):
                    if row < col:
                        with cols[col]:
                            # Select the correlation type
                            selected_corr = st.selectbox(
                                f"Correlation between {numeric_columns[row].name} and {numeric_columns[col].name}",
                                options=correlation_options,
                                index=0,  # Default to "No Correlation"
                                key=f"corr_{row}_{col}",
                            )

                            # Set correlation value
                            if selected_corr != "No Correlation":
                                # Randomly select a value within the specified range
                                corr_range = CORRELATION_TYPES.get(selected_corr, (None, None))
                                if corr_range:
                                    # Use numpy to generate a random value within the specified range
                                    correlation_value = np.random.uniform(corr_range[0], corr_range[1])
                                    correlation_matrix[row, col] = correlation_value
                                    correlation_matrix[col, row] = correlation_value

                            correlation_value_matrix[row, col] = selected_corr
                            correlation_value_matrix[col, row] = selected_corr

            # Set the diagonal to 1.0 (perfect self-correlation)
            np.fill_diagonal(correlation_matrix, 1.0)
            np.fill_diagonal(correlation_value_matrix, "Perfect Correlation")

            # Apply correlations button
            if st.button("Apply Correlations"):
                for row in range(len(numeric_columns)):
                    column = numeric_columns[row]
                    column_correlations = []
                    for col in range(len(numeric_columns)):
                        if row != col:
                            correlation_type = correlation_value_matrix[row, col]
                            correlation_value = correlation_matrix[row, col]

                            column_correlations.append({
                                "Base Column": numeric_columns[col].name,
                                "Correlation Type": correlation_type,
                                "Correlation Value": correlation_value if correlation_value is not None else "No Correlation"
                            })
                    column.correlations = column_correlations
                st.success("Correlations updated successfully!")

            # Display correlation matrix overview
            st.write("### Correlation Matrix Overview")
            
            # Prepare dataframes for both correlation types and values
            corr_type_df = pd.DataFrame(
                correlation_value_matrix,
                columns=[col.name for col in numeric_columns],
                index=[col.name for col in numeric_columns],
            )
            st.write("Correlation Types:")
            st.dataframe(corr_type_df)
            
            # Create a separate dataframe for numeric correlation values
            corr_value_df = pd.DataFrame(
                correlation_matrix,
                columns=[col.name for col in numeric_columns],
                index=[col.name for col in numeric_columns],
            )
            st.write("Correlation Numeric Values:")
            st.dataframe(corr_value_df)



    def _render_generation_section(self):
        """Enhanced dataset generation section with correlation control."""
        st.write("## Step 4: Generate Dataset")
        # Add a toggle for correlation application
        apply_correlations = st.checkbox(
            "Apply Correlations",
            help="Check to generate dataset with configured correlations",
            value=False,  # Default to not applying correlations
        )
        # Generation button
        if st.button("Generate Dataset", type="primary"):
            with st.spinner("Generating dataset..."):
                start_time = time.time()
                # Sorting columns remains the same
                # sorted_columns = self.sort_columns_by_dependencies(
                #     st.session_state.columns
                # )
                data = {}
                for column in st.session_state.columns:
                    data[column.name] = self.data_generator.generate_data(
                        column,
                        st.session_state.num_rows,
                        data if apply_correlations else None,
                    )
                st.session_state.df = pd.DataFrame(data)
                elapsed_time = time.time() - start_time
                st.success(
                    f"Dataset generated successfully in {elapsed_time:.2f} seconds!"
                )
                st.write(st.session_state.df)
                csv = st.session_state.df.to_csv(index=False)
                st.download_button(
                    "Download CSV", data=csv, file_name="generated_dataset.csv"
                )
                # Analyze data
                DataAnalyzer.analyze_dataset(st.session_state.df)


class DataAnalyzer:
    """Handles basic data analysis and visualization for generated datasets."""

    @staticmethod
    def test_normality(series: pd.Series) -> dict:
        """Perform Shapiro-Wilk normality test if sample size is sufficient."""
        if len(series) < 3:
            return None
        shapiro_stat, shapiro_p = stats.shapiro(series)
        return {
            "Shapiro-Wilk Test": {
                "statistic": shapiro_stat,
                "p-value": shapiro_p,
                "is_normal": shapiro_p > 0.05,
            }
        }

    @staticmethod
    def plot_distributions(numeric_df: pd.DataFrame) -> None:
        """Plot basic distribution visualizations for numeric columns, only when requested."""
        if not numeric_df.empty:
            st.write("### Distribution Visualizations")
            for column in numeric_df.columns:
                with st.expander(f"Analyze {column} Distribution"):
                    data = numeric_df[column]
                    # Histogram with KDE
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.histplot(data=data, kde=True, ax=ax)
                    ax.set_title(f"Histogram with KDE for {column}")
                    st.pyplot(fig)
                    plt.close()
                    # Normality Test
                    normality_result = DataAnalyzer.test_normality(data)
                    if normality_result:
                        st.write(
                            f"Normality Test Result: {'Normal' if normality_result['Shapiro-Wilk Test']['is_normal'] else 'Not Normal'} "
                            f"(p-value: {normality_result['Shapiro-Wilk Test']['p-value']:.4f})"
                        )

    @staticmethod
    def analyze_dataset(df: pd.DataFrame) -> None:
        """Perform basic analysis on the dataset."""
        st.markdown("---")
        st.write("### Dataset Summary")
        st.write(df.describe())
        numeric_df = df.select_dtypes(include=["number"])
        if not numeric_df.empty:
            DataAnalyzer.plot_distributions(numeric_df)
        else:
            st.write("No numeric columns available for analysis.")


def main():
    """Main application entry point."""
    st.set_page_config(page_title="Dataset Generator", page_icon="📊", layout="wide")
    ui = DatasetUI()
    ui.render_configuration_ui()


if __name__ == "__main__":
    main()
