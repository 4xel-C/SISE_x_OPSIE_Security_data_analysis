# SISE_x_OPSIE_Security_data_analysis

## TODO List:

- [] Confirm with OPSIE list of critical ports (parser.py file)

## Parsing

- Grouping logs by source ip
- Feature engineering:
    - access number
    - distinc ipdist
    - permit nbr
    - deny nbr
    - permit small ports
    - permit admin ports

- advanced feature engineering:
    - deny rate
    - unique_dst_ratio: unique dst numbers / nbr access => Horizontal scanning
    - unique_port_ratio: unique_port / nbr access => Vertical scanning
    - activty duration in second
    - request per second
    - distinct_rules_hit
    - deny_rules_hit
    - most trigered rule
    - sensitive ports nbr
    - sensitive ports ratio


Check LLM:

  Tier 1 — Indispensables

  ┌─────────────────────┬──────────────────────────────────────────────┐
  │       Feature       │                   Pourquoi                   │
  ├─────────────────────┼──────────────────────────────────────────────┤
  │ deny_rate           │ Signal le plus direct d'une IP bloquée       │
  ├─────────────────────┼──────────────────────────────────────────────┤
  │ unique_dst_ratio    │ Détecte le scan horizontal                   │
  ├─────────────────────┼──────────────────────────────────────────────┤
  │ unique_port_ratio   │ Détecte le scan vertical                     │
  ├─────────────────────┼──────────────────────────────────────────────┤
  │ requests_per_second │ Détecte les bursts/comportements automatisés │
  └─────────────────────┴──────────────────────────────────────────────┘

  Ces 4 résument à elles seules la majorité des patterns d'attaque connus.

  Tier 2 — Bon complément

  ┌───────────────────────┬────────────────────────────────────────────────────────────────┐
  │        Feature        │                            Pourquoi                            │
  ├───────────────────────┼────────────────────────────────────────────────────────────────┤
  │ deny_rules_hit        │ Diversité des règles en Deny — plus robuste que deny_nbr seul  │
  ├───────────────────────┼────────────────────────────────────────────────────────────────┤
  │ sensitive_ports_ratio │ Cible des services critiques                                   │
  ├───────────────────────┼────────────────────────────────────────────────────────────────┤
  │ activity_duration_s   │ Distingue une attaque courte/intense d'un trafic long/régulier │
  └───────────────────────┴────────────────────────────────────────────────────────────────┘

  Tier 3 — Redondantes pour le ML

  ┌─────────────────────────────────────────────────┬───────────────────────────────────────────────────────┐
  │                     Feature                     │                  Pourquoi redondante                  │
  ├─────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
  │ access_nbr                                      │ Capturé par requests_per_second + activity_duration_s │
  ├─────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
  │ permit_nbr / deny_nbr                           │ Capturés par deny_rate                                │
  ├─────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
  │ distinct_ipdst / distinct_portdst               │ Capturés par les ratios                               │
  ├─────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
  │ sensitive_ports_nbr                             │ Capturé par sensitive_ports_ratio                     │
  ├─────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
  │ permit_small_ports_nbr / permit_admin_ports_nbr │ Peu discriminants seuls                               │
  ├─────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
  │ distinct_rules_hit                              │ Corrélé à deny_rules_hit  