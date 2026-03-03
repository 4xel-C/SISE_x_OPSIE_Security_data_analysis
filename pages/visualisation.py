import ipaddress

import folium
import streamlit as st
import streamlit.components.v1 as components
from folium.plugins import HeatMap
from pandas import Timestamp
from streamlit_folium import st_folium

from services.charts import (
    access_distribution,
    allow_deny_pie,
    deny_permit_timeline,
    deny_rate_distribution,
    deny_rules_distribution,
    deny_vs_permit_bubble,
    horizontal_vs_vertical_scan,
    ip_rank_scatter,
    port_distribution_bar,
    proto_action_bar,
    requests_per_second_top,
    sensitive_ports_top,
    top10_permitted_ports_bar,
    top5_ip_sources_bar,
    top_dst_ports_bar,
    top_triggered_rules,
)

_UNIV_SUBNETS = [
    ipaddress.ip_network("159.84.0.0/16"),
]

_PORT_NAMES: dict[int, str] = {
    20: "FTP-Data", 21: "FTP", 22: "SSH", 23: "Telnet", 25: "SMTP",
    53: "DNS", 67: "DHCP", 68: "DHCP", 69: "TFTP", 80: "HTTP",
    110: "POP3", 123: "NTP", 139: "NetBIOS", 143: "IMAP",
    161: "SNMP", 162: "SNMPTRAP", 389: "LDAP", 443: "HTTPS",
    445: "SMB", 465: "SMTPS", 514: "Syslog", 515: "LPD",
    636: "LDAPS", 993: "IMAPS", 995: "POP3S",
}


def _is_internal(ip_str: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_str)
        return any(ip in net for net in _UNIV_SUBNETS)
    except ValueError:
        return False


st.title("Visualisation")

data = st.session_state.data

# =============================================================================
# SIDEBAR — FILTERS
# =============================================================================
st.sidebar.header("Filtres")

t_min, t_max = data.time_range

start_date, end_date = st.sidebar.slider(
    "Fenêtre temporelle",
    min_value=t_min.to_pydatetime(),
    max_value=t_max.to_pydatetime(),
    value=(t_min.to_pydatetime(), t_max.to_pydatetime()),
    format="DD/MM/YY HH:mm",
)


@st.cache_data
def get_df(start: Timestamp, end: Timestamp):
    return st.session_state.data.get_filtered_df(start, end)


@st.cache_data
def get_df_raw(start: Timestamp, end: Timestamp):
    dm = st.session_state.data
    mask = (dm.df_raw["date"] >= start) & (dm.df_raw["date"] <= end)
    return dm.df_raw[mask].copy()


# Get the filtered dataframe based on the time window.
df = get_df(Timestamp(start_date), Timestamp(end_date))


# =============================================================================
# ANALYSE DESCRIPTIVE
# =============================================================================
st.header("Analyse descriptive")

# --- Inline filters ---
df_raw = get_df_raw(Timestamp(start_date), Timestamp(end_date))

fcol1, fcol2, fcol3 = st.columns([1, 2, 1])
with fcol1:
    proto_filter = st.radio(
        "Protocole",
        options=["Tous", "TCP", "UDP"],
        horizontal=True,
    )
with fcol2:
    port_range_filter = st.selectbox(
        "Plage de ports destination",
        options=["Tous", "Clients (49152–65535)", "Autres (0–49151)"],
    )
with fcol3:
    action_filter = st.radio(
        "Action",
        options=["Tous", "Permit", "Deny"],
        horizontal=True,
    )

if proto_filter != "Tous":
    df_raw = df_raw[df_raw["proto"] == proto_filter]

if port_range_filter == "Clients (49152–65535)":
    df_raw = df_raw[df_raw["portdst"] >= 49152]
elif port_range_filter == "Autres (0–49151)":
    df_raw = df_raw[df_raw["portdst"] < 49152]

if action_filter != "Tous":
    df_raw = df_raw[df_raw["action"] == action_filter]

total_raw = len(df_raw)
permit_raw = int((df_raw["action"] == "Permit").sum())
deny_raw = int((df_raw["action"] == "Deny").sum())
tcp_raw = int((df_raw["proto"] == "TCP").sum())
udp_raw = int((df_raw["proto"] == "UDP").sum())

st.markdown(f"""
<style>
.desc-stat-grid {{
    display: flex;
    gap: 12px;
    flex-wrap: nowrap;
    margin: 16px 0 24px 0;
}}
.desc-stat-card {{
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 16px 20px;
    flex: 1;
    min-width: 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}}
.desc-stat-label {{
    font-size: 11px;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 8px;
    white-space: nowrap;
}}
.desc-stat-value {{
    font-size: 32px;
    font-weight: 700;
    line-height: 1;
}}
</style>
<div class="desc-stat-grid">
    <div class="desc-stat-card">
        <div class="desc-stat-label">Total flux</div>
        <div class="desc-stat-value" style="color:#0891b2;">{total_raw:,}</div>
    </div>
    <div class="desc-stat-card">
        <div class="desc-stat-label">✓ Autorisés</div>
        <div class="desc-stat-value" style="color:#059669;">{permit_raw:,}</div>
    </div>
    <div class="desc-stat-card">
        <div class="desc-stat-label">✗ Rejetés</div>
        <div class="desc-stat-value" style="color:#dc2626;">{deny_raw:,}</div>
    </div>
    <div class="desc-stat-card">
        <div class="desc-stat-label">TCP</div>
        <div class="desc-stat-value" style="color:#4f46e5;">{tcp_raw:,}</div>
    </div>
    <div class="desc-stat-card">
        <div class="desc-stat-label">UDP</div>
        <div class="desc-stat-value" style="color:#d97706;">{udp_raw:,}</div>
    </div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.plotly_chart(proto_action_bar(df_raw), width="stretch")
with col2:
    st.plotly_chart(allow_deny_pie(df_raw), width="stretch")
with col3:
    st.plotly_chart(port_distribution_bar(df_raw), width="stretch")

st.plotly_chart(top_dst_ports_bar(df_raw), width="stretch")

# =============================================================================
# LOG TABLE
# =============================================================================
st.header("Journal des flux")

tcol1, tcol2, tcol3, tcol4 = st.columns([3, 1, 1, 1])
with tcol1:
    search_query = st.text_input(
        "search", placeholder="🔍  Rechercher (IP, port, protocole…)",
        label_visibility="collapsed",
    )
with tcol2:
    action_tbl = st.selectbox("action", ["Toutes actions", "Permit", "Deny"],
                              label_visibility="collapsed")
with tcol3:
    proto_tbl = st.selectbox("proto", ["Tous protocoles", "TCP", "UDP"],
                             label_visibility="collapsed")
with tcol4:
    rows_per_page = st.selectbox("lignes", [15, 25, 50, 100],
                                 label_visibility="collapsed")

df_table = df_raw.copy()

if search_query:
    q = search_query.lower()
    df_table = df_table[
        df_table["ipsrc"].str.lower().str.contains(q, na=False)
        | df_table["ipdst"].str.lower().str.contains(q, na=False)
        | df_table["portdst"].astype(str).str.contains(q, na=False)
        | df_table["proto"].str.lower().str.contains(q, na=False)
        | df_table["action"].str.lower().str.contains(q, na=False)
    ]

if action_tbl != "Toutes actions":
    df_table = df_table[df_table["action"] == action_tbl]

if proto_tbl != "Tous protocoles":
    df_table = df_table[df_table["proto"] == proto_tbl]


def _log_table_html(df, n: int) -> str:
    rows_html = ""
    for _, row in df.head(n).iterrows():
        p_color = "#4f46e5" if row["proto"] == "TCP" else "#d97706"
        p_bg    = "#eef2ff" if row["proto"] == "TCP" else "#fef3c7"
        a_color = "#059669" if row["action"] == "Permit" else "#dc2626"
        a_bg    = "#d1fae5" if row["action"] == "Permit" else "#fee2e2"
        ts = row["date"].strftime("%H:%M:%S") if hasattr(row["date"], "strftime") else row["date"]
        rows_html += f"""
        <tr>
            <td>{ts}</td>
            <td><b>{row['ipsrc']}</b></td>
            <td>{row['ipdst']}</td>
            <td style="color:#0891b2;font-weight:600;">{row['portdst']}</td>
            <td><span style="background:{p_bg};color:{p_color};border:1px solid {p_color}44;
                padding:2px 10px;border-radius:20px;font-size:11px;font-weight:700;">
                {row['proto']}</span></td>
            <td><span style="background:{a_bg};color:{a_color};border:1px solid {a_color}44;
                padding:2px 10px;border-radius:20px;font-size:11px;font-weight:700;">
                {row['action'].upper()}</span></td>
            <td style="color:#94a3b8;font-size:12px;">{row['regle']}</td>
        </tr>"""
    return f"""
    <style>
    .log-wrap{{border:1px solid #e2e8f0;border-radius:10px;overflow:hidden;}}
    .log-scroll{{max-height:480px;overflow-y:auto;}}
    .log-scroll::-webkit-scrollbar{{width:6px;}}
    .log-scroll::-webkit-scrollbar-track{{background:#f1f5f9;}}
    .log-scroll::-webkit-scrollbar-thumb{{background:#cbd5e1;border-radius:3px;}}
    .log-table{{width:100%;border-collapse:collapse;font-family:'JetBrains Mono',monospace;font-size:13px;}}
    .log-table thead tr{{position:sticky;top:0;z-index:1;}}
    .log-table th{{background:#f8fafc;color:#64748b;font-size:11px;text-transform:uppercase;
        letter-spacing:.08em;padding:10px 12px;text-align:left;border-bottom:2px solid #e2e8f0;}}
    .log-table td{{padding:10px 12px;border-bottom:1px solid #f1f5f9;color:#334155;vertical-align:middle;}}
    .log-table tr:hover td{{background:#f8fafc;}}
    </style>
    <p style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#64748b;margin:0 0 8px 2px;">
        {len(df):,} résultats</p>
    <div class="log-wrap">
      <div class="log-scroll">
        <table class="log-table">
            <thead><tr>
                <th>Timestamp</th><th>IP Source</th><th>IP Destination</th>
                <th>Port DST</th><th>Proto</th><th>Action</th><th>Règle</th>
            </tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
      </div>
    </div>"""


st.markdown(_log_table_html(df_table, rows_per_page), unsafe_allow_html=True)

# =============================================================================
# TAB 3: INTERACTIVE IP VISUALIZATION
# =============================================================================
st.header("Visualisation interactive des IP Sources")

ip_agg = (
    df_raw.groupby("ipsrc")
    .agg(
        total_flux=("ipsrc", "count"),
        distinct_dst=("ipdst", "nunique"),
        permit=("action", lambda x: (x == "Permit").sum()),
        deny=("action", lambda x: (x == "Deny").sum()),
    )
    .reset_index()
    .sort_values("total_flux", ascending=False)
    .reset_index(drop=True)
)

if ip_agg.empty:
    st.warning("Aucune IP source trouvée pour la fenêtre temporelle et les filtres sélectionnés.")
else:
    ip_list = ip_agg["ipsrc"].tolist()

    # Resolve current IP from session state
    current_ip = st.session_state.get("selected_ip")
    if current_ip not in ip_list:
        current_ip = ip_list[0]

    # If a chart click was stored from the previous run, apply it now (before slider renders)
    # by deleting the slider key so value= takes effect instead of cached state.
    if "pending_ip_click" in st.session_state:
        desired_rank = st.session_state.pop("pending_ip_click")
        st.session_state.pop("ip_rank_slider", None)
    else:
        desired_rank = ip_list.index(current_ip) + 1

    ip_rank = st.slider(
        "Se balader dans les IPs",
        min_value=1,
        max_value=len(ip_agg),
        value=desired_rank,
        key="ip_rank_slider",
    )

    selected_ip = ip_agg["ipsrc"].iloc[ip_rank - 1]
    st.session_state["selected_ip"] = selected_ip

    detail_col, scatter_col = st.columns([1, 2])

    with scatter_col:
        scatter_sel = st.plotly_chart(
            ip_rank_scatter(ip_agg, selected_ip),
            width="stretch",
            on_select="rerun",
            key="ip_scatter_rank",
        )

    # Chart click → store pending rank, rerun (slider key reset happens at top of next run)
    sel_points = getattr(getattr(scatter_sel, "selection", None), "points", []) or []
    if sel_points:
        cd = sel_points[0].get("customdata", [])
        if cd and cd[0] in ip_list:
            clicked_ip = cd[0]
            if clicked_ip != selected_ip:
                st.session_state["selected_ip"] = clicked_ip
                st.session_state["pending_ip_click"] = ip_list.index(clicked_ip) + 1
                st.rerun()

    # --- Detail panel (left column, beside scatter) ---
    ip_row = ip_agg[ip_agg["ipsrc"] == selected_ip].iloc[0]
    dst_agg = (
        df_raw[df_raw["ipsrc"] == selected_ip]
        .groupby("ipdst")
        .agg(
            permit=("action", lambda x: (x == "Permit").sum()),
            deny=("action", lambda x: (x == "Deny").sum()),
        )
        .reset_index()
        .sort_values("deny", ascending=False)
    )
    dst_rows = ""
    for _, drow in dst_agg.iterrows():
        ac = "#059669" if drow["permit"] > 0 else "#94a3b8"
        dc = "#dc2626" if drow["deny"] > 0 else "#94a3b8"
        dst_rows += f"""
    <div style="display:flex;justify-content:space-between;align-items:center;
                padding:7px 4px;border-bottom:1px solid #f1f5f9;">
        <span style="font-size:13px;color:#334155;">{drow['ipdst']}</span>
        <div style="font-size:12px;display:flex;gap:14px;">
            <span style="color:{ac};">✓{int(drow['permit'])}</span>
            <span style="color:{dc};">✗{int(drow['deny'])}</span>
        </div>
    </div>"""

    no_data = '<div style="padding:20px;text-align:center;color:#94a3b8;font-size:13px;">Aucune donnée</div>'
    with detail_col:
        components.html(f"""
    <html><head>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet">
    </head><body style="margin:0;padding:0;background:transparent;font-family:'JetBrains Mono',monospace;">
    <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:10px;
                padding:16px;box-shadow:0 1px 3px rgba(0,0,0,0.06);">
        <div style="font-size:15px;font-weight:700;color:#0f172a;margin-bottom:2px;">
            Détail : {selected_ip}</div>
        <div style="font-size:11px;color:#64748b;margin-bottom:12px;">
            {int(ip_row['total_flux'])} flux · {int(ip_row['distinct_dst'])} destinations</div>
        <div style="display:flex;gap:8px;margin-bottom:16px;">
            <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;
                        padding:10px 14px;flex:1;">
                <div style="font-size:10px;color:#16a34a;text-transform:uppercase;
                            letter-spacing:.08em;margin-bottom:4px;">✓ Autorisés</div>
                <div style="font-size:26px;font-weight:700;color:#059669;">{int(ip_row['permit'])}</div>
            </div>
            <div style="background:#fef2f2;border:1px solid #fecaca;border-radius:8px;
                        padding:10px 14px;flex:1;">
                <div style="font-size:10px;color:#dc2626;text-transform:uppercase;
                            letter-spacing:.08em;margin-bottom:4px;">✗ Rejetés</div>
                <div style="font-size:26px;font-weight:700;color:#dc2626;">{int(ip_row['deny'])}</div>
            </div>
        </div>
        <div style="font-size:10px;color:#64748b;text-transform:uppercase;
                    letter-spacing:.08em;margin-bottom:6px;">Destinations contactées</div>
        <div style="max-height:255px;overflow-y:auto;border:1px solid #e2e8f0;
                    border-radius:8px;padding:0 10px;background:#f8fafc;">
            {dst_rows if dst_rows else no_data}
        </div>
    </div>
    </body></html>""", height=492, scrolling=False)

# =============================================================================
# TOP 5 IP SOURCES
# =============================================================================
st.header("TOP 5 des IP Sources les plus émettrices")

_top5_agg = (
    df_raw.groupby("ipsrc")
    .agg(
        permit=("action", lambda x: (x == "Permit").sum()),
        deny=("action", lambda x: (x == "Deny").sum()),
    )
    .reset_index()
)
_top5_agg["total"] = _top5_agg["permit"] + _top5_agg["deny"]
_top5_agg["deny_rate"] = _top5_agg["deny"] / _top5_agg["total"]
_top5_data = _top5_agg.nlargest(5, "total").reset_index(drop=True)

_RANK_COLORS = ["#fbbf24", "#94a3b8", "#cd7c2e", "#64748b", "#475569"]
_rank_rows = ""
for _i, _row in _top5_data.iterrows():
    _rc = _RANK_COLORS[_i] if _i < len(_RANK_COLORS) else "#475569"
    _deny_pct = int(_row["deny_rate"] * 100)
    _ext_badge = (
        ""
        if _is_internal(_row["ipsrc"])
        else (
            '<span style="margin-left:6px;background:#f1f5f9;color:#64748b;'
            'border:1px solid #e2e8f0;padding:1px 6px;border-radius:10px;font-size:10px;"> Externe</span>'
        )
    )
    _rank_rows += f"""
    <div style="display:flex;align-items:center;gap:12px;padding:10px 8px;
                border-bottom:1px solid #f1f5f9;">
        <div style="min-width:28px;height:28px;border-radius:50%;background:{_rc};
                    display:flex;align-items:center;justify-content:center;
                    font-weight:700;font-size:13px;color:#ffffff;">{_i + 1}</div>
        <div style="flex:1;min-width:0;">
            <div style="font-weight:700;font-size:14px;color:#0f172a;">{_row['ipsrc']}</div>
            <div style="font-size:11px;color:#64748b;margin-top:2px;">
                {int(_row['total'])} flux{_ext_badge}</div>
        </div>
        <div style="text-align:right;font-size:12px;white-space:nowrap;">
            <span style="color:#059669;">✓{int(_row['permit'])}</span>
            <span style="color:#dc2626;margin-left:8px;">✗{int(_row['deny'])}</span>
            <div style="font-size:11px;color:#94a3b8;margin-top:2px;">{_deny_pct}% rejeté</div>
        </div>
    </div>"""

_top5_chart_col, _top5_list_col = st.columns([3, 2])
with _top5_chart_col:
    st.plotly_chart(top5_ip_sources_bar(df_raw), width="stretch")
with _top5_list_col:
    components.html(f"""
    <html><head>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet">
    </head><body style="margin:0;padding:0;background:transparent;font-family:'JetBrains Mono',monospace;">
    <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:10px;
                padding:0 8px;box-shadow:0 1px 3px rgba(0,0,0,0.06);">
        {_rank_rows}
    </div>
    </body></html>""", height=340)

# =============================================================================
# TOP 10 PORTS < 1024
# =============================================================================
st.header("TOP 10 des ports inférieurs à 1024 (accès autorisés)")

_permitted_ports = df_raw[(df_raw["action"] == "Permit") & (df_raw["portdst"] < 1024)]
_port_counts = _permitted_ports["portdst"].value_counts().head(10).reset_index()
_port_counts.columns = ["port", "count"]

_p10_chart_col, _p10_tags_col = st.columns([2, 1])
with _p10_chart_col:
    st.plotly_chart(top10_permitted_ports_bar(df_raw), width="stretch")

_TAG_COLORS = [
    "#38bdf8", "#34d399", "#f87171", "#fbbf24", "#a78bfa",
    "#f472b6", "#4ade80", "#fb923c", "#60a5fa", "#e879f9",
]
_tags_html = ""
for _idx, (_, _pr) in enumerate(_port_counts.iterrows()):
    _pname = _PORT_NAMES.get(int(_pr["port"]), "")
    _pname_str = f" {_pname}" if _pname else ""
    _c = _TAG_COLORS[_idx % len(_TAG_COLORS)]
    _tags_html += (
        f'<span style="background:#f8fafc;border:1px solid {_c}66;color:{_c};'
        f'padding:5px 12px;border-radius:20px;font-size:12px;font-weight:600;white-space:nowrap;">'
        f':{int(_pr["port"])}{_pname_str} ({int(_pr["count"])})</span>'
    )

with _p10_tags_col:
    components.html(f"""
    <html><head>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet">
    </head><body style="margin:0;padding:8px 0;background:transparent;font-family:'JetBrains Mono',monospace;">
    <div style="display:flex;flex-wrap:wrap;gap:8px;align-content:flex-start;">{_tags_html}</div>
    </body></html>""", height=310)

# =============================================================================
# EXTERNAL IPs (hors plan d'adressage universitaire)
# =============================================================================
st.header("Accès hors plan d'adressage universitaire")

_ext_mask = ~df_raw["ipsrc"].apply(_is_internal)
_df_ext = df_raw[_ext_mask]
_ext_agg = (
    _df_ext.groupby("ipsrc")
    .agg(
        flux=("ipsrc", "count"),
        permit=("action", lambda x: (x == "Permit").sum()),
        deny=("action", lambda x: (x == "Deny").sum()),
        distinct_dst=("ipdst", "nunique"),
        ports=("portdst", lambda x: ", ".join(
            str(p) for p in x.value_counts().head(3).index
        )),
    )
    .reset_index()
    .sort_values("flux", ascending=False)
)

_subnet_str = ", ".join(str(s) for s in _UNIV_SUBNETS)
st.caption(f"{len(_ext_agg)} IP externes détectées · Subnets univ : {_subnet_str}")

_ext_rows = ""
for _, _er in _ext_agg.iterrows():
    _ext_rows += f"""
    <tr>
        <td style="color:#0891b2;font-weight:700;padding:9px 12px;">{_er['ipsrc']}</td>
        <td style="color:#334155;text-align:center;padding:9px 8px;">{int(_er['flux'])}</td>
        <td style="text-align:center;padding:9px 8px;">
            <span style="background:#f0fdf4;color:#059669;border:1px solid #bbf7d0;
                         padding:2px 10px;border-radius:20px;font-weight:700;">{int(_er['permit'])}</span></td>
        <td style="text-align:center;padding:9px 8px;">
            <span style="background:#fef2f2;color:#dc2626;border:1px solid #fecaca;
                         padding:2px 10px;border-radius:20px;font-weight:700;">{int(_er['deny'])}</span></td>
        <td style="color:#64748b;text-align:center;padding:9px 8px;">{int(_er['distinct_dst'])}</td>
        <td style="color:#94a3b8;font-size:11px;padding:9px 12px;">{_er['ports']}</td>
    </tr>"""

components.html(f"""
<html><head>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet">
</head><body style="margin:0;padding:0;background:transparent;font-family:'JetBrains Mono',monospace;">
<div style="border:1px solid #e2e8f0;border-radius:10px;overflow:hidden;
            box-shadow:0 1px 3px rgba(0,0,0,0.06);">
<div style="max-height:420px;overflow-y:auto;">
<table style="width:100%;border-collapse:collapse;font-size:13px;">
<thead><tr>
    <th style="background:#f8fafc;color:#64748b;font-size:10px;text-transform:uppercase;
               letter-spacing:.08em;padding:10px 12px;text-align:left;
               border-bottom:2px solid #e2e8f0;position:sticky;top:0;">IP Source</th>
    <th style="background:#f8fafc;color:#64748b;font-size:10px;text-transform:uppercase;
               letter-spacing:.08em;padding:10px 8px;text-align:center;
               border-bottom:2px solid #e2e8f0;position:sticky;top:0;">Flux</th>
    <th style="background:#f8fafc;color:#64748b;font-size:10px;text-transform:uppercase;
               letter-spacing:.08em;padding:10px 8px;text-align:center;
               border-bottom:2px solid #e2e8f0;position:sticky;top:0;">Autorisés</th>
    <th style="background:#f8fafc;color:#64748b;font-size:10px;text-transform:uppercase;
               letter-spacing:.08em;padding:10px 8px;text-align:center;
               border-bottom:2px solid #e2e8f0;position:sticky;top:0;">Rejetés</th>
    <th style="background:#f8fafc;color:#64748b;font-size:10px;text-transform:uppercase;
               letter-spacing:.08em;padding:10px 8px;text-align:center;
               border-bottom:2px solid #e2e8f0;position:sticky;top:0;">Dest.</th>
    <th style="background:#f8fafc;color:#64748b;font-size:10px;text-transform:uppercase;
               letter-spacing:.08em;padding:10px 12px;text-align:left;
               border-bottom:2px solid #e2e8f0;position:sticky;top:0;">Ports fréquents</th>
</tr></thead>
<tbody style="background:#ffffff;">{_ext_rows}</tbody>
</table>
</div></div>
</body></html>""", height=460)

# =============================================================================
# GEO HEATMAP
# =============================================================================
st.header("Carte de géolocalisation des IP sources")


@st.fragment
def _render_geo_heatmap():
    # Compute flux per IP from the currently filtered df_raw (respects all active filters)
    ip_flux = df_raw.groupby("ipsrc").size().reset_index(name="access_nbr_filtered")

    # Get geo info from the aggregated df (has lat/lon) and keep only IPs in filtered data
    geo_full = df.reset_index().dropna(subset=["lat", "lon"])
    geo_df = geo_full.merge(ip_flux, on="ipsrc", how="inner")

    if geo_df.empty:
        st.info("Aucune donnée de géolocalisation disponible (toutes les IPs sont internes ou non résolues).")
        return

    city_heat = (
        geo_df.dropna(subset=["city"])
        .groupby(["city", "country", "lat", "lon"])
        .agg(flux=("access_nbr_filtered", "sum"), ips=("ipsrc", "count"))
        .reset_index()
    )

    map_col, stat_col = st.columns([3, 1])

    with stat_col:
        st.metric("Pays détectés", city_heat["country"].nunique())
        st.metric("Villes détectées", len(city_heat))
        st.metric("IPs géolocalisées", len(geo_df))
        st.dataframe(
            city_heat[["city", "country", "flux"]]
            .sort_values("flux", ascending=False)
            .reset_index(drop=True),
            width="stretch",
            height=300,
        )

    with map_col:
        m = folium.Map(location=[20, 0], zoom_start=2, tiles="CartoDB positron")

        heat_data = [
            [row["lat"], row["lon"], row["flux"]]
            for _, row in city_heat.iterrows()
        ]
        HeatMap(heat_data, radius=30, blur=20, max_zoom=8, min_opacity=0.4).add_to(m)

        for _, cr in city_heat.iterrows():
            folium.CircleMarker(
                location=[cr["lat"], cr["lon"]],
                radius=5,
                color="#0891b2",
                fill=True,
                fill_color="#0891b2",
                fill_opacity=0.7,
                tooltip=(
                    f"<b>{cr['city']}</b>, {cr['country']}<br>"
                    f"{int(cr['flux'])} flux · {int(cr['ips'])} IP(s)"
                ),
            ).add_to(m)

        st_folium(m, width="stretch", height=460)


_render_geo_heatmap()

# =============================================================================
# DENY / PERMIT TIMELINE
# =============================================================================
st.header("Évolution temporelle des flux Deny / Permit")
st.plotly_chart(deny_permit_timeline(df_raw), width="stretch")

# =============================================================================
# IP INDICATORS TABLE
# =============================================================================
st.header("Tableau des indicateurs par IP source")


@st.fragment
def _render_ip_indicators_table():
    _itcol1, _itcol2, _itcol3, _itcol4 = st.columns([3, 1, 1, 1])
    with _itcol1:
        _ip_search = st.text_input(
            "ip_tbl_search", placeholder="🔍  Rechercher une IP source…",
            label_visibility="collapsed",
        )
    with _itcol2:
        _deny_filter = st.selectbox(
            "ip_tbl_deny", ["Tous", "Deny ≥ 80%", "Deny ≥ 50%", "Deny = 0%"],
            label_visibility="collapsed",
        )
    with _itcol3:
        _sort_col = st.selectbox(
            "ip_tbl_sort", ["Flux total", "Deny rate", "Req/s", "Ports sensibles"],
            label_visibility="collapsed",
        )
    with _itcol4:
        _ip_rows = st.selectbox(
            "ip_tbl_rows", [25, 50, 100, 200],
            label_visibility="collapsed",
        )

    _df_iptbl = df.reset_index().copy()

    if _ip_search:
        _df_iptbl = _df_iptbl[_df_iptbl["ipsrc"].str.contains(_ip_search, na=False)]

    if _deny_filter == "Deny ≥ 80%":
        _df_iptbl = _df_iptbl[_df_iptbl["deny_rate"] >= 0.8]
    elif _deny_filter == "Deny ≥ 50%":
        _df_iptbl = _df_iptbl[_df_iptbl["deny_rate"] >= 0.5]
    elif _deny_filter == "Deny = 0%":
        _df_iptbl = _df_iptbl[_df_iptbl["deny_rate"] == 0]

    _sort_map = {
        "Flux total": "access_nbr",
        "Deny rate": "deny_rate",
        "Req/s": "requests_per_second",
        "Ports sensibles": "sensitive_ports_nbr",
    }
    _df_iptbl = _df_iptbl.sort_values(_sort_map[_sort_col], ascending=False)

    st.markdown(_ip_indicators_html(_df_iptbl, _ip_rows), unsafe_allow_html=True)


def _ip_indicators_html(df_ip, n: int) -> str:
    rows_html = ""
    for _, row in df_ip.head(n).iterrows():
        dr = row["deny_rate"]
        if dr >= 0.8:
            dr_color, dr_bg = "#dc2626", "#fef2f2"
        elif dr >= 0.5:
            dr_color, dr_bg = "#d97706", "#fef3c7"
        else:
            dr_color, dr_bg = "#059669", "#f0fdf4"

        sp = int(row["sensitive_ports_nbr"])
        sp_color = "#dc2626" if sp > 0 else "#94a3b8"
        sp_weight = "700" if sp > 0 else "400"

        rows_html += f"""
        <tr>
            <td><b style="color:#0891b2;">{row['ipsrc']}</b></td>
            <td style="text-align:right;">{int(row['access_nbr']):,}</td>
            <td style="text-align:right;">{int(row['distinct_ipdst'])}</td>
            <td style="text-align:right;">{int(row['distinct_portdst'])}</td>
            <td style="text-align:right;color:#059669;">{int(row['permit_nbr']):,}</td>
            <td style="text-align:right;color:#dc2626;">{int(row['deny_nbr']):,}</td>
            <td style="text-align:center;">
                <span style="background:{dr_bg};color:{dr_color};border:1px solid {dr_color}44;
                    padding:2px 8px;border-radius:20px;font-size:11px;font-weight:700;">
                    {dr:.0%}</span></td>
            <td style="text-align:right;color:#64748b;">{row['unique_dst_ratio']:.3f}</td>
            <td style="text-align:right;color:#64748b;">{row['unique_port_ratio']:.3f}</td>
            <td style="text-align:right;color:#64748b;">{row['activity_duration_s']:.0f}s</td>
            <td style="text-align:right;color:#64748b;">{row['requests_per_second']:.2f}</td>
            <td style="text-align:right;">{int(row['distinct_rules_hit'])}</td>
            <td style="text-align:right;color:#dc2626;">{int(row['deny_rules_hit'])}</td>
            <td style="color:#94a3b8;font-size:11px;max-width:130px;
                overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{row['most_triggered_rule']}</td>
            <td style="text-align:right;color:{sp_color};font-weight:{sp_weight};">{sp}</td>
            <td style="text-align:right;color:#64748b;">{row['sensitive_ports_ratio']:.1%}</td>
            <td style="text-align:right;color:#059669;">{int(row['permit_small_ports_nbr']):,}</td>
            <td style="text-align:right;color:#818cf8;">{int(row['permit_admin_ports_nbr']):,}</td>
        </tr>"""

    headers = [
        "IP Source", "Flux", "Dst uniq.", "Ports uniq.",
        "Autorisés", "Rejetés", "Deny rate",
        "Ratio dst", "Ratio ports", "Durée", "Req/s",
        "Règles", "Règles deny", "Règle principale",
        "Ports sens.", "Ratio sens.", "Sys &lt;1024", "Appli 1024+",
    ]
    th_html = "".join(f"<th>{h}</th>" for h in headers)

    return f"""
    <style>
    .ip-wrap{{border:1px solid #e2e8f0;border-radius:10px;overflow:hidden;}}
    .ip-scroll{{max-height:500px;overflow-y:auto;overflow-x:auto;}}
    .ip-scroll::-webkit-scrollbar{{width:6px;height:6px;}}
    .ip-scroll::-webkit-scrollbar-track{{background:#f1f5f9;}}
    .ip-scroll::-webkit-scrollbar-thumb{{background:#cbd5e1;border-radius:3px;}}
    .ip-table{{width:100%;border-collapse:collapse;font-family:'JetBrains Mono',monospace;font-size:12px;white-space:nowrap;}}
    .ip-table thead tr{{position:sticky;top:0;z-index:1;}}
    .ip-table th{{background:#f8fafc;color:#64748b;font-size:10px;text-transform:uppercase;
        letter-spacing:.06em;padding:9px 10px;text-align:left;border-bottom:2px solid #e2e8f0;}}
    .ip-table td{{padding:8px 10px;border-bottom:1px solid #f1f5f9;color:#334155;vertical-align:middle;}}
    .ip-table tr:hover td{{background:#f8fafc;}}
    </style>
    <p style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#64748b;margin:0 0 8px 2px;">
        {len(df_ip):,} IP sources</p>
    <div class="ip-wrap">
      <div class="ip-scroll">
        <table class="ip-table">
            <thead><tr>{th_html}</tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
      </div>
    </div>"""


_render_ip_indicators_table()

# =============================================================================
# IMPRESSION
# =============================================================================
st.header("Impression de l'analyse")

st.caption(
    f"Fenêtre temporelle : {start_date.strftime('%d/%m/%Y %H:%M')} — {end_date.strftime('%d/%m/%Y %H:%M')}"
)

# Inject print CSS into the parent page (not inside the iframe)
st.markdown("""
<style>
@media print {
    section[data-testid="stSidebar"],
    header[data-testid="stHeader"],
    div[data-testid="stToolbar"],
    div[data-testid="stDecoration"],
    div[data-testid="stStatusWidget"] { display: none !important; }
    .main .block-container { padding: 0 !important; max-width: 100% !important; }
    iframe { display: none !important; }
}
</style>
""", unsafe_allow_html=True)

components.html("""
<style>
.print-btn {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #0f172a;
    color: #ffffff;
    border: none;
    border-radius: 8px;
    padding: 12px 28px;
    font-size: 14px;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    cursor: pointer;
    letter-spacing: 0.04em;
    transition: background 0.15s;
}
.print-btn:hover { background: #1e293b; }
</style>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@600&display=swap" rel="stylesheet">
<button class="print-btn" onclick="window.parent.print()">
    &#128438;&nbsp; Imprimer / Exporter en PDF
</button>
""", height=70)
