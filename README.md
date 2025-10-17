# Prop Stats

Single collection of all useful stats about the props in the BF6 Portal SDK. No need to download or run this script, just grab the [latest json here](https://github.com/The-Sir-Community/prop_stats/releases/latest/download/prop_stats.json).

## Query the Data

Install [duckDB]()

```sql
SELECT 
    name,
    path,
    category,
    physicsCost,
    bounding_box_volume,
    footprint,
    height,
    volume,
    volume_ratio,
    is_watertight,
    triangle_count,
    is_potentially_invalid,
    levelRestrictions
FROM read_json('https://github.com/The-Sir-Community/prop_stats/releases/latest/download/prop_stats.json')
WHERE list_contains(levelRestrictions, 'MP_Battery')
ORDER BY name;
```

## Instructions:

1. `uv sync`
2. `uv run main.py C:\Users\user\PortalSDK\GodotProject\raw\models --asset-types C:\Users\user\PortalSDK\FbExportData\asset_types.json`