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

app = FastAPI(title="FLUX Virtual Try-On API")

# Load pipeline global
transformer = FluxTransformer2DModel.from_pretrained(
    "xiaozaa/catvton-flux-beta",  torch_dtype=torch.bfloat16
)
pipe = FluxFillPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    transformer=transformer,
    torch_dtype=torch.bfloat16
).to("cuda")
pipe.transformer.to(torch.bfloat16)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
mask_transform = transforms.Compose([transforms.ToTensor()])

UPLOAD_DIR = "./uploads"
POSE_DIR = "./assets/poses"
MASK_DIR = "./assets/masks"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def run_inference(image_path, mask_path, garment_path, size=(576, 768),
                  num_steps=50, guidance_scale=30, seed=42):
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

    # 🔥 Xuất file PNG (không nén, giữ chất lượng cao)
    img_bytes = io.BytesIO()
    tryon_result.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    return img_bytes

def describe_image(garment: UploadFile = File(...)):
    # đọc bytes từ UploadFile
    image_bytes = garment.file.read()
    garment.file.seek(0)

    # tạo client
    client = genai.Client(api_key="")

    # nội dung gửi vào: ảnh + prompt
    contents = [
        types.Part.from_bytes(
            data=image_bytes,
            mime_type=garment.content_type or "image/jpeg",
        ),
        """
You are an image classification AI. Analyze the input image and identify the type of outfit shown. 
Return ONLY one label in lowercase from the following fixed set:

- "upper" → if the main item is a top (shirt, t-shirt, jacket, hoodie, sweater, blouse, crop-top, etc.)
- "lower" → if the main item is a bottom (pants, jeans, shorts, skirt, leggings, etc.)
- "full" → if the outfit is a one-piece (dress, jumpsuit, bodysuit, long coat covering whole body, etc.)
- "full" → if the image does not clearly show an outfit or does not fit the above categories.

Your response must contain ONLY one word: upper, lower, full, or other.
"""
    ]

    # gọi API
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents
    )

    print(response.text)
    return response.text

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
    print("Using mask:", mask_path)
    
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

    # Trả PNG chất lượng cao
    return StreamingResponse(img_bytes, media_type="image/png")