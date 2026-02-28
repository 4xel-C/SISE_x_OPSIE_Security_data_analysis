import pandas as pd
from pandas import DataFrame


class parser:
    """Singleton class to parse the raw data"""

    _instance = None

    def __new__(cls, filename):
        """Singleton pattern to ensure only one instance of the parser is created"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.__init__(filename)
        return cls._instance

    def __init__(self, filename):
        if not filename.endswith(".csv"):
            raise ValueError("The file must be a csv file")

        self.filename = filename

        self.df_raw = pd.read_csv(self.filename)

        # set the columns
        self.df_raw.columns = [
            "ipsrc",
            "ipdst",
            "portdst",
            "proto",
            "action",
            "date",
            "regle",
        ]

        self.df_raw["date"] = pd.to_datetime(self.df_raw["date"])

        self._aggregate_ip()
        self._feature_engineering()

    def _aggregate_ip(self) -> DataFrame:
        """Aggregate the raw dataframe by ipsrc and set up all the metrics properly

        Returns:
            DataFrame: The aggregated dataframe with all the metrics
        """

        groups = self.df_raw.groupby("ipsrc")

        df = groups.agg(
            access_nbr=("ipsrc", "count"),
            distinct_ipdst=("ipdst", "nunique"),
            distinct_portdst=("portdst", "nunique"),
            permit_nbr=("action", lambda x: sum(x == "Permit")),
            deny_nbr=("action", lambda x: sum(x == "Deny")),
        )

        # get the number of permit actions with small ports (portdst < 1024)
        n_permit_small_ports = groups.apply(
            lambda x: sum((x["action"] == "Permit") & (x["portdst"] < 1024))
        )

        n_permit_admin_ports = groups.apply(
            lambda x: sum(
                (x["action"] == "Permit")
                & (x["portdst"] < 49152)
                & (x["portdst"] >= 1024)
            )
        )

        df = df.join(n_permit_small_ports.rename("permit_small_ports_nbr"))
        df = df.join(n_permit_admin_ports.rename("permit_admin_ports_nbr"))

        self.df = df

        return df

    def _feature_engineering(self) -> None:
        """Augment self.df with derived features for visualization and ML.

        Adds:
        - Ratios: deny_rate, unique_dst_ratio, unique_port_ratio, sensitive_ports_ratio
        - Temporal: activity_duration_s, requests_per_second
        - Rules: distinct_rules_hit, deny_rules_hit, most_triggered_rule
        - Sensitive ports: sensitive_ports_nbr, sensitive_ports_ratio
        """

        # Sensitive ports (SSH, Telnet, SMTP, DNS, SMB, MySQL, RDP, common admin ports)
        # TODO: To be checked by OPSIE
        SENSITIVE_PORTS = {22, 23, 25, 53, 445, 3306, 3389, 8080, 8443}

        groups = self.df_raw.groupby("ipsrc")

        # --- Ratios ---
        self.df["deny_rate"] = self.df["deny_nbr"] / self.df["access_nbr"]

        # horizontal scanning indicatores (many ipdst)
        self.df["unique_dst_ratio"] = self.df["distinct_ipdst"] / self.df["access_nbr"]

        # vertical scanning indicators (many portdst)
        self.df["unique_port_ratio"] = (
            self.df["distinct_portdst"] / self.df["access_nbr"]
        )

        # --- Temporal ---
        temporal = groups["date"].agg(first_seen="min", last_seen="max")

        # duration in seconds
        self.df["activity_duration_s"] = (
            (temporal["last_seen"] - temporal["first_seen"])
            .dt.total_seconds()
            .clip(lower=1)  # to avoid division by zero if activity duration < 1s
        )

        self.df["requests_per_second"] = (
            self.df["access_nbr"] / self.df["activity_duration_s"]
        )

        # --- Rules ---
        self.df["distinct_rules_hit"] = groups["regle"].nunique()
        self.df["deny_rules_hit"] = (
            self.df_raw[self.df_raw["action"] == "Deny"]
            .groupby("ipsrc")["regle"]
            .nunique()
            .reindex(self.df.index, fill_value=0)  # to fill ipsrc with no deny action
        )
        self.df["most_triggered_rule"] = groups["regle"].agg(
            lambda x: x.value_counts().idxmax()
        )

        # --- Sensitive ports ---
        self.df["sensitive_ports_nbr"] = groups.apply(
            lambda x: sum(x["portdst"].isin(SENSITIVE_PORTS))
        )
        self.df["sensitive_ports_ratio"] = (
            self.df["sensitive_ports_nbr"] / self.df["access_nbr"]
        )
