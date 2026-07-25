import base64
from pathlib import Path
import fitz  # PyMuPDF
from openai import OpenAI

from backend.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL

# Vision Multimodal Language Model
MULTIMODAL_MODEL = "nvidia/nemotron-nano-12b-v2-vl:free"

# Initialize OpenRouter / OpenAI client instance
ai_client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
)


def extract_pdf_embedded_images(file_path: str, destination_dir: str = "data/uploads/images") -> list[dict]:
    """Extracts embedded graphical assets from PDF pages and saves them to disk."""
    output_path = Path(destination_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    document = fitz.open(file_path)
    extracted_records = []

    for idx, page in enumerate(document, start=1):
        image_metadata = page.get_images(full=True)

        for img_idx, img_info in enumerate(image_metadata, start=1):
            x_ref = img_info[0]
            raw_img = document.extract_image(x_ref)

            img_bytes = raw_img["image"]
            extension = raw_img["ext"]

            save_name = f"doc_p{idx}_img{img_idx}.{extension}"
            full_save_path = output_path / save_name

            with open(full_save_path, "wb") as img_file:
                img_file.write(img_bytes)

            extracted_records.append({
                "filepath": str(full_save_path),
                "page_num": idx,
            })

    document.close()
    return extracted_records


def convert_file_to_base64(filepath: str) -> str:
    """Encodes a local file into a Base64 format string."""
    with open(filepath, "rb") as target_file:
        return base64.b64encode(target_file.read()).decode("utf-8")


def generate_image_caption(image_location: str) -> str:
    """Sends image payload to Vision LLM to synthesize structural and textual descriptions."""
    b64_data = convert_file_to_base64(image_location)
    file_ext = Path(image_location).suffix.lstrip(".")

    system_prompt = (
        "Please provide a comprehensive structural breakdown of this image. "
        "Extract all embedded text, tabulated data, figures, and numerical values explicitly."
    )

    completion = ai_client.chat.completions.create(
        model=MULTIMODAL_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": system_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{file_ext};base64,{b64_data}"
                        },
                    },
                ],
            }
        ],
    )
    return completion.choices[0].message.content


def parse_and_caption_pdf_images(pdf_file_path: str) -> list[dict]:
    """Orchestrates image extraction and automated textual description for PDF visual content."""
    image_entries = extract_pdf_embedded_images(pdf_file_path)
    processed_results = []

    for item in image_entries:
        try:
            caption_text = generate_image_caption(item["filepath"])
            processed_results.append({
                "page": item["page_num"],
                "content": caption_text,
                "source_type": "image",
                "image_path": item["filepath"],
            })
        except Exception as err:
            print(f"[Vision Processing Error] Failed to caption {item['filepath']}: {err}")

    return processed_results