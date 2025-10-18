from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Iterable

from openai import OpenAI

EXAMPLE_JSON = {
    "name": "CommandPost_01_PropsC",
    "bounding_box": {
      "x": {
        "min": -4.48569,
        "max": 5.54836
      },
      "y": {
        "min": -0.0,
        "max": 3.82646
      },
      "z": {
        "min": -5.60797,
        "max": 5.51292
      }
    },
    "bounding_box_volume": 426.98536,
    "footprint": 111.58757,
    "height": 3.82646,
    "volume": 144.76405,
    "volume_ratio": 0.33904,
    "center_of_mass": {
      "x": 0.07325,
      "y": 4.54609,
      "z": 0.00682
    },
    "is_watertight": False,
    "triangle_count": 6591,
    "is_potentially_invalid": False,
    "path": "Uncategorized",
    "physicsCost": 46,
    "category": "spatial",
    "levelRestrictions": [
      "MP_Capstone"
    ]
},

EXAMPLE_IMAGE = "data:image/png;base64,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"


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate AI descriptions for GLB assets using vision models."
    )
    parser.add_argument(
        "stats_json",
        type=Path,
        help="Path to the stats JSON file generated by main.py.",
    )
    parser.add_argument(
        "thumbnails_directory",
        type=Path,
        help="Folder containing PNG thumbnails (named matching the GLB files).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Path to write the enhanced JSON file. Defaults to <stats_json>_with_descriptions.json.",
    )
    parser.add_argument(
        "-k",
        "--api-key",
        type=str,
        help="OpenRouter API key. Can also be set via OPENROUTER_API_KEY env var.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="mistralai/mistral-small-3.2-24b-instruct:free",
        help="Model to use via OpenRouter (default: mistralai/mistral-small-3.2-24b-instruct:free).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip items that already have a description field.",
    )
    return parser.parse_args(argv)


def encode_image_to_base64(image_path: Path) -> str:
    """Encode an image file to base64 string."""
    with image_path.open("rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def build_prompt(asset_data: dict) -> str:
    """Build the prompt for the vision model."""
    # Remove description field if present to avoid feeding it to the model
    asset_info = {k: v for k, v in asset_data.items() if k != "description"}

    prompt = f"""<asset_data>
{json.dumps(asset_info, indent=2)}
</asset_data>

<instructions>
Based on the untextured thumbnail image and the technical specifications above, provide a concise, descriptive summary of this 3D asset. Include:

1. What the object appears to be (e.g., "industrial oil silo", "concrete rubble debris", "military AA gun")
2. Key visual characteristics (style, condition)
3. Any notable features or details visible in the thumbnail

These thumbnails show untextured models, so do not describe colors or surface textures.
Do not comment on the grey appearance since these models are untextured. Many details might also be lower resolution in these thumbnails, such as a potted plant which may appear grey with blocky leaves, while in-game they are rendered with greater detail.
Do not summarize the info in the json, this is already available. Only use the JSON data to help understand what the object is.
Do not include the level that it is available in.
Do not add any commentary that is not a description of the object.

Keep the description to 1-2 sentences, suitable for a game asset database.

Respond with ONLY the description text, no additional formatting or preamble.
</instructions>
"""

    return prompt


def generate_description(
    asset_data: dict,
    thumbnail_path: Path,
    client: OpenAI,
    model: str,
) -> tuple[str | None, dict]:
    """
    Generate a description for an asset using OpenRouter's vision API.

    Returns a tuple of (description text or None, usage_info dict).
    usage_info contains: cost (float), prompt_tokens (int), completion_tokens (int), total_tokens (int)
    """
    if not thumbnail_path.exists():
        print(f"  Warning: Thumbnail not found at {thumbnail_path}")
        return None, {"cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    # Encode image
    try:
        image_base64 = encode_image_to_base64(thumbnail_path)
    except Exception as exc:
        print(f"  Error encoding image: {exc}")
        return None, {"cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    # Build prompt
    prompt = build_prompt(asset_data)

    # Make API request using OpenAI client
    try:
        # Build example context with JSON
        example_context = f"""<example>
<example_asset_data>
{json.dumps(EXAMPLE_JSON, indent=2)}
</example_asset_data>

<example_output>
Rectangular military field structure resembling a deployable command tent or operations shelter, with rigid frame walls, angled roof sections, ventilation ducts, and external stairs leading to an entry flap.
</example_output>
</example>"""

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": example_context,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": EXAMPLE_IMAGE
                            },
                        },
                        {
                            "type": "text",
                            "text": "Now analyze this asset:",
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            },
                        },
                    ],
                }
            ],
        )

        # Extract description from response
        description = response.choices[0].message.content.strip()

        # Extract usage information from response
        usage_info = {
            "cost": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }

        if hasattr(response, 'usage') and response.usage:
            # Extract token counts
            if hasattr(response.usage, 'prompt_tokens'):
                usage_info["prompt_tokens"] = response.usage.prompt_tokens
            if hasattr(response.usage, 'completion_tokens'):
                usage_info["completion_tokens"] = response.usage.completion_tokens
            if hasattr(response.usage, 'total_tokens'):
                usage_info["total_tokens"] = response.usage.total_tokens

            # Extract cost (OpenRouter provides this in usage metadata)
            if hasattr(response.usage, 'total_cost'):
                usage_info["cost"] = float(response.usage.total_cost)

        return description, usage_info

    except Exception as exc:
        print(f"  API request failed: {exc}")
        return None, {"cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)

    thumbnails_directory = args.thumbnails_directory.expanduser().resolve()
    if not thumbnails_directory.is_dir():
        raise SystemExit(f"Thumbnails directory not found: {thumbnails_directory}")

    stats_json_path = args.stats_json.expanduser().resolve()
    if not stats_json_path.is_file():
        raise SystemExit(f"Stats JSON file not found: {stats_json_path}")

    # Determine output path
    output_path = args.output
    if output_path is None:
        output_path = stats_json_path.parent / f"{stats_json_path.stem}_with_descriptions.json"
    output_path = output_path.expanduser().resolve()

    # Get API key
    api_key = args.api_key
    if not api_key:
        import os
        api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit(
            "OpenRouter API key required. Provide via --api-key or OPENROUTER_API_KEY env var."
        )

    # Initialize OpenAI client configured for OpenRouter
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    # Load stats JSON
    print(f"Loading stats from {stats_json_path}")
    with stats_json_path.open("r", encoding="utf-8") as f:
        stats_data = json.load(f)

    if not isinstance(stats_data, list):
        raise SystemExit("Expected stats JSON to contain a list of objects")

    print(f"Loaded {len(stats_data)} assets")
    print(f"Using model: {args.model}")

    # Process each asset
    processed_count = 0
    skipped_count = 0
    failed_count = 0
    total_cost = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    for index, asset_data in enumerate(stats_data, start=1):
        asset_name = asset_data.get("name", f"unknown_{index}")
        print(f"[{index}/{len(stats_data)}] Processing {asset_name}...")

        # Skip if description already exists and --skip-existing is set
        if args.skip_existing and "description" in asset_data:
            print(f"  Skipping (description already exists)")
            skipped_count += 1
            continue

        # Find thumbnail
        thumbnail_path = thumbnails_directory / f"{asset_name}.png"

        # Generate description
        description, usage_info = generate_description(
            asset_data,
            thumbnail_path,
            client,
            args.model,
        )

        if description:
            asset_data["description"] = description

            # Accumulate usage statistics
            total_cost += usage_info["cost"]
            total_prompt_tokens += usage_info["prompt_tokens"]
            total_completion_tokens += usage_info["completion_tokens"]
            total_tokens += usage_info["total_tokens"]

            print(f"  Generated: {description[:80]}...")
            if usage_info["cost"] > 0 or usage_info["total_tokens"] > 0:
                print(f"  Tokens: {usage_info['prompt_tokens']} prompt + {usage_info['completion_tokens']} completion = {usage_info['total_tokens']} total")
                if usage_info["cost"] > 0:
                    print(f"  Cost: ${usage_info['cost']:.6f}")
            processed_count += 1
        else:
            print(f"  Failed to generate description")
            failed_count += 1

    # Write enhanced JSON
    print(f"\nWriting enhanced data to {output_path}")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(stats_data, f, indent=2)

    # Calculate and display statistics
    avg_cost = total_cost / processed_count if processed_count > 0 else 0.0
    avg_prompt_tokens = total_prompt_tokens / processed_count if processed_count > 0 else 0.0
    avg_completion_tokens = total_completion_tokens / processed_count if processed_count > 0 else 0.0
    avg_total_tokens = total_tokens / processed_count if processed_count > 0 else 0.0

    print(f"\nComplete!")
    print(f"  Processed: {processed_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Total: {len(stats_data)}")
    print(f"\nToken Usage:")
    print(f"  Total prompt tokens: {total_prompt_tokens:,}")
    print(f"  Total completion tokens: {total_completion_tokens:,}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Average prompt tokens per item: {avg_prompt_tokens:.1f}")
    print(f"  Average completion tokens per item: {avg_completion_tokens:.1f}")
    print(f"  Average total tokens per item: {avg_total_tokens:.1f}")
    print(f"\nCost Summary:")
    print(f"  Total cost: ${total_cost:.6f}")
    print(f"  Average cost per item: ${avg_cost:.6f}")


if __name__ == "__main__":
    main(sys.argv[1:])
