# Imports
from datetime import date
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Union, Optional
import numpy as np
import pandas as pd
import random
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
from scipy.linalg import cholesky, LinAlgError
import seaborn as sns
import streamlit as st
from faker import Faker
import time

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
        use_normal = params.get("Use Normal Distribution", False)
        if use_normal:
            mean, std = (
                float(params.get("Mean", (min_val + max_val) / 2)),
                float(params.get("Standard Deviation", (max_val - min_val) / 6)),
            )
            values = np.clip(
                np.round(np.random.normal(mean, std, num_rows)), min_val, max_val
            ).astype(int)
            return pd.Series(values)
        return pd.Series(np.random.randint(min_val, max_val, num_rows))

    def _generate_float(self, params: ParamDict, num_rows: int) -> pd.Series:
        min_val, max_val, decimals = (
            params.get("Min", 0.0),
            params.get("Max", 100.0),
            params.get("Decimals", 2),
        )
        mean, use_normal = (
            float(params.get("Mean", (min_val + max_val) / 2)),
            params.get("Use Normal Distribution", False),
        )
        if use_normal:
            std = float(params.get("Standard Deviation", (max_val - min_val) / 6))
            return pd.Series(
                np.round(
                    np.clip(np.random.normal(mean, std, num_rows), min_val, max_val),
                    decimals,
                )
            )
        return pd.Series(
            np.round(np.random.uniform(min_val, max_val, num_rows), decimals)
        )

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

    def _generate_custom_list(self, params: ParamDict, num_rows: int) -> pd.Series:
        """
        Generate a custom list based on user-defined values and their corresponding percentages.
        Args:
            params (ParamDict): A dictionary containing "Values" and "Percentages".
            num_rows (int): Number of rows in the generated dataset.
        Returns:
            pd.Series: A Pandas Series with randomly chosen values based on percentages.
        """
        # Extract values and percentages
        values = params["Value Names"]
        percentages = params["Percentages"]
        # Ensure percentages sum to 100 and values/percentages match
        if not np.isclose(sum(percentages), 100.0):
            raise ValueError("Percentages must sum to 100.")
        if len(values) != len(percentages):
            raise ValueError("The number of values and percentages must match.")
        # Convert percentages to probabilities
        probabilities = np.array(percentages) / 100
        # Generate random choices
        choices = np.random.choice(values, num_rows, p=probabilities)
        return pd.Series(choices)

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
        # Ensure stability (positive semi-definite)
        correlation_matrix = np.maximum(correlation_matrix, correlation_matrix.T)
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
            col3, col4 = st.columns(2)
            with col3:
                # Minimum value input
                min_val = (
                    -1000000.0
                    if data_type
                    in [DataType.FLOAT, DataType.CURRENCY, DataType.PERCENTAGE]
                    else -1000000
                )
                params["Min"] = self._create_number_input(
                    "Minimum", min_val, 1000000.0, 0.0, 0.1, f"min_{idx}"
                )
            with col4:
                # Maximum value input
                max_val = (
                    1000000.0
                    if data_type
                    in [DataType.FLOAT, DataType.CURRENCY, DataType.PERCENTAGE]
                    else 1000000
                )
                params["Max"] = self._create_number_input(
                    "Maximum", min_val, 1000000.0, 100.0, 0.1, f"max_{idx}"
                )
        # Decimal places for float
        if data_type == DataType.FLOAT:
            decimal_places = st.number_input(
                "Number of Decimal Places",
                min_value=1,
                max_value=10,
                value=2,
                key=f"decimals_{idx}",
            )
            params["Decimals"] = decimal_places
        # Normal distribution option
        if data_type in [DataType.INTEGER, DataType.FLOAT]:
            use_normal = self._create_checkbox(
                "Use Normal Distribution",
                key=f"use_normal_{idx}",
                help_text="Generate data following a normal distribution",
            )
            params["Use Normal Distribution"] = use_normal
            if use_normal:
                col1, col2 = st.columns(2)
                with col1:
                    params["Mean"] = self._create_number_input(
                        "Mean", -1000000.0, 1000000.0, 50.0, 0.1, f"mean_{idx}"
                    )
                with col2:
                    params["Standard Deviation"] = self._create_number_input(
                        "Standard Deviation", 0.1, 1000000.0, 10.0, 0.1, f"std_{idx}"
                    )
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
                value=3,  # Default for demonstration
                step=1,
                key=f"num_values_{idx}",
            )
            params["Num Values"] = num_values
            # Initialize lists for names and percentages
            value_names = [None] * int(num_values)
            percentages = [None] * int(
                num_values
            )  # Start with None for unset percentages

            # Define remaining percentage logic
            def distribute_remaining(percentages):
                total_assigned = sum(p for p in percentages if p is not None)
                remaining = max(0.0, 100.0 - total_assigned)  # Remaining percentage
                num_unset = len([p for p in percentages if p is None])
                # Distribute remaining percentage among unset values
                auto_fill = remaining / num_unset if num_unset > 0 else 0.0
                for i, p in enumerate(percentages):
                    if p is None:
                        percentages[i] = auto_fill
                return auto_fill

            # Dynamic inputs for names and percentages
            st.write("### Define Values and Percentages")
            remaining_label = st.empty()
            for i in range(int(num_values)):
                col1, col2 = st.columns([2, 1])  # Name gets more space
                with col1:
                    value_names[i] = st.text_input(
                        f"Value {i + 1} Name",
                        value=f"Value_{i + 1}",  # Default name
                        key=f"value_name_{idx}_{i}",
                    )
                with col2:
                    # Input for percentage with live redistribution
                    percentage = st.number_input(
                        f"Value {i + 1} %",
                        min_value=0.0,
                        max_value=100.0,
                        step=1.0,
                        key=f"percentage_{idx}_{i}",
                        format="%.2f",
                    )
                    percentages[i] = percentage if percentage > 0 else None
            # Calculate auto-filled percentage and update display
            auto_fill_value = distribute_remaining(percentages)
            remaining_label.markdown(
                f"**Remaining percentage auto-filled for unset values: {auto_fill_value:.2f}%**"
                if auto_fill_value > 0
                else "**All percentages set manually.**"
            )
            # Display validation feedback
            if sum(percentages) != 100.0:
                st.warning(
                    f"Total percentage must equal 100%. Currently: {sum(percentages):.2f}%"
                )
            else:
                st.success(f"Total percentage is valid: {sum(percentages):.2f}%")
                # Save params
            params["Value Names"] = value_names
            params["Percentages"] = percentages
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
            if col.type
            in [
                DataType.INTEGER,
                DataType.FLOAT,
                DataType.PERCENTAGE,
                # Add more types as needed
            ]
        ]
        if len(numeric_columns) < 2:
            st.info("Add at least two numeric columns to configure correlations.")
            return
        with st.container(border=True):
            st.write("### Correlation Matrix Configuration")
            st.markdown("""
            - Select correlation types between numeric columns
            - 🟢 **Positive Correlation**: Variables move together
            - 🔴 **Negative Correlation**: Variables move in opposite directions
            - 🔷 **No Correlation**: Variables are independent
            """)
            # Initialize correlation matrix
            correlation_matrix = np.full(
                (len(numeric_columns), len(numeric_columns)), "No Correlation"
            )
            correlation_options = ["No Correlation"] + list(
                self.CORRELATION_TYPES.keys()
            )
            # Create a grid of selectboxes for each pair of columns
            for row in range(len(numeric_columns)):
                cols = st.columns(len(numeric_columns))
                for col in range(len(numeric_columns)):
                    if row < col:
                        # Display the selectbox for each correlation pair in a grid layout
                        with cols[col]:
                            # Select the correlation type from the dropdown
                            selected_corr = st.selectbox(
                                f"Correlation between {numeric_columns[row].name} and {numeric_columns[col].name}",
                                options=correlation_options,
                                index=correlation_options.index(
                                    correlation_matrix[row, col]
                                ),
                                key=f"corr_{row}_{col}",
                            )
                            # Set correlation value in the matrix
                            correlation_matrix[row, col] = selected_corr
                            correlation_matrix[col, row] = selected_corr
            # Apply correlations button
            if st.button("Apply Correlations"):
                for row in range(len(numeric_columns)):
                    column = numeric_columns[row]
                    column_correlations = [
                        {
                            "Base Column": numeric_columns[col].name,
                            "Correlation Value": correlation_matrix[row, col],
                        }
                        for col in range(len(numeric_columns))
                        if row != col
                    ]
                    column.correlations = column_correlations
                st.success("Correlations updated successfully!")
            # Display correlation matrix overview
            st.write("### Correlation Matrix Overview")
            corr_df = pd.DataFrame(
                correlation_matrix,
                columns=[col.name for col in numeric_columns],
                index=[col.name for col in numeric_columns],
            )
            st.dataframe(corr_df)

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
