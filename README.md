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

### Generate Statistics

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Generate GLB statistics:
   ```bash
   uv run main.py C:\Users\user\PortalSDK\GodotProject\raw\models --asset-types C:\Users\user\PortalSDK\FbExportData\asset_types.json
   ```

### Generate AI Descriptions

Generate natural language descriptions for assets using vision AI:

```bash
# Set your OpenRouter API key
export OPENROUTER_API_KEY="your-api-key-here"

# Generate descriptions
uv run generate_descriptions.py \
  C:\Users\user\PortalSDK\GodotProject\raw\models \
  glb_stats.json \
  C:\Users\user\PortalSDK\Thumbnails
```

**Options:**
- `-o, --output`: Output path for enhanced JSON (default: `<input>_with_descriptions.json`)
- `-k, --api-key`: OpenRouter API key (or use `OPENROUTER_API_KEY` env var)
- `-m, --model`: Model to use (default: `anthropic/claude-3.5-sonnet`)
- `--skip-existing`: Skip items that already have descriptions

**Example with options:**
```bash
uv run generate_descriptions.py \
  ./models \
  glb_stats.json \
  ./thumbnails \
  --output prop_stats_full.json \
  --model anthropic/claude-3.5-sonnet \
  --skip-existing
```

**Supported Models:**
- `anthropic/claude-3.5-sonnet` (recommended for quality)
- `anthropic/claude-3-haiku` (faster/cheaper)
- `openai/gpt-4-vision-preview`
- See [OpenRouter models](https://openrouter.ai/models) for more options