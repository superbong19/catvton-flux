from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
import shutil
import io
import os
import torch
from torchvision import transforms
from diffusers import FluxFillPipeline, FluxTransformer2DModel
from diffusers.utils import load_image
from google import genai
from google.genai import types

# =========================
# FastAPI app
# =========================
app = FastAPI(title="FLUX Virtual Try-On API")

# =========================
# Directories (robust lookup)
# =========================
# The app may be run in different environments (local, Modal container, etc.).
# Try several candidate base paths and pick the first that contains an `assets/` folder.
candidate_bases = [
    os.environ.get("WORKDIR", "/root/app"),
    os.path.dirname(__file__),
    os.getcwd(),
    "/root/app2",
    "/root",
    "/workspace",
    "/home",
]

APP_BASE = None
for base in candidate_bases:
    try:
        if base and os.path.isdir(os.path.join(base, "assets")):
            APP_BASE = base
            break
    except Exception:
        continue

if APP_BASE is None:
    # fallback to WORKDIR env or /root/app
    APP_BASE = os.environ.get("WORKDIR", "/root/app")

WORKDIR = APP_BASE
UPLOAD_DIR = os.path.join(WORKDIR, "uploads")
POSE_DIR = os.path.join(WORKDIR, "assets/poses")
MASK_DIR = os.path.join(WORKDIR, "assets/masks")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Helpful debug info in logs about where assets are loaded from
print(f"[main] WORKDIR={WORKDIR}")
print(f"[main] POSE_DIR={POSE_DIR}")
print(f"[main] MASK_DIR={MASK_DIR}")

# =========================
# Transforms
# =========================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
mask_transform = transforms.Compose([transforms.ToTensor()])

# =========================
# Lazy-load FLUX pipeline
# =========================
pipe = None
transformer = None

def get_pipe():
    global pipe, transformer
    if pipe is None:
        transformer = FluxTransformer2DModel.from_pretrained(
            "xiaozaa/catvton-flux-beta", torch_dtype=torch.bfloat16
        )
        pipe = FluxFillPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            transformer=transformer,
            torch_dtype=torch.bfloat16
        ).to("cuda")
        pipe.transformer.to(torch.bfloat16)
    return pipe

# =========================
# Run inference
# =========================
def run_inference(image_path, mask_path, garment_path, size=(576, 768),
                  num_steps=50, guidance_scale=30, seed=42):

    pipe = get_pipe()

    image = load_image(image_path).convert("RGB").resize(size)
    mask = load_image(mask_path).convert("RGB").resize(size)
    garment = load_image(garment_path).convert("RGB").resize(size)

    image_tensor = transform(image)
    mask_tensor = mask_transform(mask)[:1]
    garment_tensor = transform(garment)

    inpaint_image = torch.cat([garment_tensor, image_tensor], dim=2)
    garment_mask = torch.zeros_like(mask_tensor)
    extended_mask = torch.cat([garment_mask, mask_tensor], dim=2)

    prompt = (
        "The pair of images highlights clothing and its styling on a model, "
        "high resolution, 4K, 8K; "
        "[IMAGE1] Detailed product shot of clothing "
        "[IMAGE2] The same cloth is worn by a model in a lifestyle setting."
    )

    generator = torch.Generator(device="cuda").manual_seed(seed)

    result = pipe(
        height=size[1],
        width=size[0] * 2,
        image=inpaint_image,
        mask_image=extended_mask,
        num_inference_steps=num_steps,
        generator=generator,
        max_sequence_length=512,
        guidance_scale=guidance_scale,
        prompt=prompt,
    ).images[0]

    width = size[0]
    tryon_result = result.crop((width, 0, width * 2, size[1]))

    img_bytes = io.BytesIO()
    tryon_result.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    return img_bytes

# =========================
# Describe image with GenAI
# =========================
def describe_image(garment: UploadFile = File(...)):
    image_bytes = garment.file.read()
    garment.file.seek(0)

    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY", ""))

    contents = [
        types.Part.from_bytes(
            data=image_bytes,
            mime_type=garment.content_type or "image/jpeg",
        ),
        """
You are an image classification AI. Analyze the input image and identify the type of outfit shown. 
Return ONLY one label in lowercase from the following fixed set:

- "upper" → if the main item is a top
- "lower" → if the main item is a bottom
- "full" → if the outfit is a one-piece
- "other" → if the image does not clearly show an outfit

Your response must contain ONLY one word: upper, lower, full, or other.
"""
    ]

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents
    )

    return response.text.strip()

# =========================
# API Endpoint
# =========================
@app.post("/tryon/")
async def virtual_tryon(
    garment: UploadFile = File(...),
    pose_id: int = Form(...),
    seed: int = Form(42),
    steps: int = Form(30),
    width: int = Form(576),
    height: int = Form(768),
):

    if not (1 <= pose_id <= 12):
        raise HTTPException(status_code=400, detail="pose_id must be between 1 and 12")

    image_path = os.path.join(POSE_DIR, f"pose-{pose_id}.jpeg")
    mask_type = describe_image(garment)
    mask_path = os.path.join(MASK_DIR, f"{mask_type}/mask-{pose_id}.png")

    if not os.path.exists(image_path) or not os.path.exists(mask_path):
        raise HTTPException(status_code=404, detail="Pose or mask file not found")

    garment_path = os.path.join(UPLOAD_DIR, garment.filename)
    with open(garment_path, "wb") as f:
        shutil.copyfileobj(garment.file, f)

    img_bytes = run_inference(
        image_path=image_path,
        mask_path=mask_path,
        garment_path=garment_path,
        size=(width, height),
        num_steps=steps,
        seed=seed
    )

    return StreamingResponse(img_bytes, media_type="image/png")
