import time

import pandas as pd
import requests
from pandas import DataFrame


class Parser:
    """Class to parse raw data and perform feature engineering for security analysis"""

    def generate_aggregated_data(self, df_raw: DataFrame) -> DataFrame:
        """Aggregate a raw dataframe by ipsrc and compute all the metrics.

        Args:
            df_raw (DataFrame): The raw dataframe to be aggregated.

        Returns:
            DataFrame: The aggregated dataframe with all the metrics
        """
        df = self._aggregate_ip(df_raw)
        df = self._feature_engineering(df, df_raw)
        df = self._geolocate_ips(df)

        return df

    def _aggregate_ip(self, df_raw: DataFrame) -> DataFrame:
        """Aggregate the raw dataframe by ipsrc and set up all the metrics properly
        Args:
            df_raw (DataFrame): The raw dataframe to be aggregated.
        Returns:
            DataFrame: The aggregated dataframe with all the metrics
        """

        groups = df_raw.groupby("ipsrc")

        df = groups[["ipsrc", "ipdst", "portdst", "action"]].agg(
            access_nbr=("ipsrc", "count"),
            distinct_ipdst=("ipdst", "nunique"),
            distinct_portdst=("portdst", "nunique"),
            permit_nbr=("action", lambda x: sum(x == "Permit")),
            deny_nbr=("action", lambda x: sum(x == "Deny")),
        )

        # TODO: Delete the  small port numbers and/admin ports to use the ports given by OPSIE team.
        # get the number of permit actions with small ports (portdst < 1024)
        n_permit_small_ports = groups[["action", "portdst"]].apply(
            lambda x: sum((x["action"] == "Permit") & (x["portdst"] < 1024))
        )

        n_permit_admin_ports = groups[["action", "portdst"]].apply(
            lambda x: sum(
                (x["action"] == "Permit")
                & (x["portdst"] < 49152)
                & (x["portdst"] >= 1024)
            )
        )

        df = df.join(n_permit_small_ports.rename("permit_small_ports_nbr"))
        df = df.join(n_permit_admin_ports.rename("permit_admin_ports_nbr"))

        return df

    def _feature_engineering(self, df: DataFrame, df_raw: DataFrame) -> DataFrame:
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

        groups = df_raw.groupby("ipsrc")

        # --- Ratios ---
        df["deny_rate"] = df["deny_nbr"] / df["access_nbr"]

        # horizontal scanning indicatores (many ipdst)
        df["unique_dst_ratio"] = df["distinct_ipdst"] / df["access_nbr"]

        # vertical scanning indicators (many portdst)
        df["unique_port_ratio"] = df["distinct_portdst"] / df["access_nbr"]

        # --- Temporal ---
        temporal = groups["date"].agg(first_seen="min", last_seen="max")

        # duration in seconds
        df["activity_duration_s"] = (
            (temporal["last_seen"] - temporal["first_seen"])
            .dt.total_seconds()
            .clip(lower=1)  # to avoid division by zero if activity duration < 1s
        )

        df["requests_per_second"] = df["access_nbr"] / df["activity_duration_s"]

        # --- Rules ---
        df["distinct_rules_hit"] = groups["regle"].nunique()
        df["deny_rules_hit"] = (
            df_raw[df_raw["action"] == "Deny"]
            .groupby("ipsrc")["regle"]
            .nunique()
            .reindex(df.index, fill_value=0)  # to fill ipsrc with no deny action
        )
        df["most_triggered_rule"] = groups["regle"].agg(
            lambda x: x.value_counts().idxmax()
        )

        # --- Sensitive ports ---
        # TODO: Add sensitive ports "common" and sensitive ports "admin"
        df["sensitive_ports_nbr"] = groups[["portdst"]].apply(
            lambda x: sum(x["portdst"].isin(SENSITIVE_PORTS))
        )
        df["sensitive_ports_ratio"] = df["sensitive_ports_nbr"] / df["access_nbr"]

        return df

    def _geolocate_ips(self, df: DataFrame) -> DataFrame:
        """Add city, country, lat, lon columns by querying the ip-api.com batch API.

        Runs once at startup (DataManager singleton). IPs that cannot be resolved
        (private, reserved, or API failure) will have NaN values.
        """
        ips = df.index.tolist()
        geo: dict[str, dict] = {}

        for i in range(0, len(ips), 100):
            batch = ips[i : i + 100]
            try:
                resp = requests.post(
                    "http://ip-api.com/batch",
                    json=[
                        {"query": ip, "fields": "query,status,city,country,lat,lon"}
                        for ip in batch
                    ],
                    timeout=10,
                )
                for item in resp.json():
                    if item.get("status") == "success":
                        geo[item["query"]] = item
            except Exception:
                pass
            # Respect free-tier rate limit: 45 requests/min
            if i + 100 < len(ips):
                time.sleep(1.2)

        df["city"]    = df.index.map(lambda ip: geo.get(ip, {}).get("city"))
        df["country"] = df.index.map(lambda ip: geo.get(ip, {}).get("country"))
        df["lat"]     = pd.to_numeric(
            df.index.map(lambda ip: geo.get(ip, {}).get("lat")), errors="coerce"
        )
        df["lon"]     = pd.to_numeric(
            df.index.map(lambda ip: geo.get(ip, {}).get("lon")), errors="coerce"
        )

        return df
