"""
Counter Manufacturing Quality Assurance System
Powered by HP ZGX Nano AI Station

A Vision Language Model demo for cosmetics manufacturing defect detection
with severity classification and event-driven alert generation.

Uses Salesforce BLIP-2 for visual inspection with
hybrid template + LLM analysis for QA reports.
"""

import os
import io
import random
import string
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timezone
from pathlib import Path
import warnings
import logging
from typing import Optional

warnings.filterwarnings('ignore')

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 8000))

# Email configuration (optional - for demo purposes)
SMTP_HOST = os.environ.get("SMTP_HOST", "")
SMTP_PORT = int(os.environ.get("SMTP_PORT", 587))
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASS = os.environ.get("SMTP_PASS", "")
ALERT_EMAIL = os.environ.get("ALERT_EMAIL", "production.lead@counter.com")

# Product lines with inspection parameters
PRODUCT_LINES = {
    "skincare_serum": {
        "name": "Skincare Serums",
        "products": ["All Bright C Serum", "Countertime Tripeptide Radiance Serum", "Overnight Resurfacing Peel"],
        "inspection_points": ["Fill level", "Color consistency", "Particulate matter", "Cap seal", "Label alignment"],
        "acceptable_color_range": "Clear to light amber",
        "container_type": "Glass dropper bottle"
    },
    "moisturizer": {
        "name": "Moisturizers & Creams",
        "products": ["Adaptive Moisture Lotion", "Countermatch Recovery Sleeping Cream", "Lotus Glow Cleansing Balm"],
        "inspection_points": ["Texture uniformity", "Color consistency", "Surface contamination", "Fill level", "Jar seal"],
        "acceptable_color_range": "White to light cream",
        "container_type": "Glass jar / Tube"
    },
    "lip_products": {
        "name": "Lip Products",
        "products": ["Sheer Lipstick", "Beyond Gloss", "Lip Conditioner"],
        "inspection_points": ["Color consistency", "Surface finish", "Tip integrity", "Component assembly", "Cap closure"],
        "acceptable_color_range": "Per SKU specification",
        "container_type": "Lipstick tube / Gloss applicator"
    },
    "mascara": {
        "name": "Mascara & Eye Products",
        "products": ["Lengthening Mascara", "Think Big All-In-One Mascara", "Lid Star Eyeshadow"],
        "inspection_points": ["Wand integrity", "Formula consistency", "Tube seal", "Label placement", "Cap threading"],
        "acceptable_color_range": "Black / Brown per SKU",
        "container_type": "Mascara tube"
    },
    "body_care": {
        "name": "Body Care",
        "products": ["Body Butter", "Citrus Mimosa Body Wash", "Hand Cream"],
        "inspection_points": ["Fill level", "Texture uniformity", "Pump mechanism", "Color consistency", "Label alignment"],
        "acceptable_color_range": "White to light cream / Clear",
        "container_type": "Pump bottle / Tube"
    },
    "sunscreen": {
        "name": "Sun Protection",
        "products": ["Countersun Mineral Sunscreen Lotion SPF 30", "Dew Skin Tinted Moisturizer SPF 20"],
        "inspection_points": ["Color consistency", "Texture uniformity", "Fill level", "Separation check", "UV filter distribution"],
        "acceptable_color_range": "White to tinted per SKU",
        "container_type": "Tube / Pump bottle"
    }
}

# Defect severity definitions
SEVERITY_LEVELS = {
    "CRITICAL": {
        "color": "#dc2626",
        "action": "STOP LINE - Immediate escalation required",
        "response_time": "Immediate",
        "examples": ["Foreign object contamination", "Microbial growth indicators", "Wrong product in container", "Allergen cross-contamination"]
    },
    "MAJOR": {
        "color": "#f59e0b",
        "action": "QUARANTINE - Hold for QA review",
        "response_time": "Within 1 hour",
        "examples": ["Significant discoloration", "Visible separation", "Incorrect fill level >10%", "Damaged seal"]
    },
    "MINOR": {
        "color": "#3b82f6",
        "action": "FLAG - Document and monitor",
        "response_time": "End of shift",
        "examples": ["Slight color variation", "Minor label misalignment", "Cosmetic packaging scuff", "Minor fill variation <5%"]
    },
    "PASS": {
        "color": "#22c55e",
        "action": "RELEASE - Approved for distribution",
        "response_time": "N/A",
        "examples": ["All inspection points within specification"]
    }
}

# Hybrid Template System for Corrective Actions
# Templates have {llm_detail} placeholder for LLM-generated content
CORRECTIVE_ACTION_TEMPLATES = {
    "CRITICAL": {
        "template": "IMMEDIATE: Stop production line and isolate batch {batch_id}. {llm_detail} Escalate to QA Manager and Plant Supervisor. Initiate root cause investigation per SOP-QA-001.",
        "fallback_detail": "Quarantine all affected units and secure evidence."
    },
    "MAJOR": {
        "template": "Remove defective unit from line and place in quarantine hold area with red tag. {llm_detail} Notify Line Supervisor within 1 hour. Review last 50 units for similar defects.",
        "fallback_detail": "Document defect with photographs and record batch/lot numbers."
    },
    "MINOR": {
        "template": "Tag unit for QA review. {llm_detail} Log in production database and continue monitoring. Report at end-of-shift QA briefing.",
        "fallback_detail": "Note variation in inspection log for trend analysis."
    }
}

ROOT_CAUSE_TEMPLATES = {
    "broken": {
        "template": "Investigate mechanical damage: {llm_detail} Check conveyor transfer points, filling station alignment, and packaging equipment gripper pressure.",
        "fallback_detail": "Review handling procedures at each production stage."
    },
    "contamination": {
        "template": "Investigate contamination source: {llm_detail} Review cleaning logs, air filtration system, and raw material COAs.",
        "fallback_detail": "Inspect operator PPE compliance and environmental controls."
    },
    "color": {
        "template": "Investigate color variation: {llm_detail} Verify batch mixing parameters and raw material lot numbers.",
        "fallback_detail": "Check colorant dispensing calibration and mixing times."
    },
    "fill": {
        "template": "Investigate fill level deviation: {llm_detail} Check filling nozzle and pump calibration.",
        "fallback_detail": "Review product viscosity and inspect level sensors."
    },
    "deformed": {
        "template": "Investigate product deformation: {llm_detail} Check storage temperature logs and mold tooling.",
        "fallback_detail": "Verify cooling cycle times and handling procedures."
    },
    "default": {
        "template": "Investigate production anomaly: {llm_detail} Pull batch records and review equipment maintenance logs.",
        "fallback_detail": "Check incoming material certifications and interview line operators."
    }
}

app = FastAPI(
    title="Counter Manufacturing QA System",
    description="VLM-powered defect detection for cosmetics manufacturing",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files from frontend directory
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")
else:
    frontend_path_alt = Path(__file__).parent / "frontend"
    if frontend_path_alt.exists():
        app.mount("/static", StaticFiles(directory=frontend_path_alt), name="static")

# Global model references
blip_processor = None
blip_model = None
text_generator = None
device = "cpu"


def load_models():
    """Load the VLM and text generation models."""
    global blip_processor, blip_model, text_generator, device
    
    if blip_model is not None:
        return
    
    from transformers import Blip2Processor, Blip2ForConditionalGeneration, pipeline
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load BLIP-2 model for image understanding
    logger.info("Loading BLIP-2 FLAN-T5-XL visual inspection model...")
    try:
        blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
        blip_model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-flan-t5-xl",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        logger.info("BLIP-2 FLAN-T5-XL model loaded successfully on " + device)
    except Exception as e:
        logger.error(f"Failed to load BLIP-2 model: {e}")
        raise
    
    # Load text generation model for report enhancement
    logger.info("Loading text generation model...")
    try:
        text_generator = pipeline(
            "text-generation",
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            device=0 if device == "cuda" else -1,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        logger.info("Text generation model loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load text generator: {e}")
        text_generator = None


def generate_inspection_id() -> str:
    """Generate a realistic inspection identifier."""
    prefix = "QA"
    line_code = random.choice(["A", "B", "C", "D"])
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    seq = f"{random.randint(1, 9999):04d}"
    return f"{prefix}-{line_code}-{date_str}-{seq}"


def generate_batch_id() -> str:
    """Generate a synthetic batch identifier."""
    year = datetime.now().strftime("%y")
    day_of_year = datetime.now().strftime("%j")
    batch_seq = f"{random.randint(1, 99):02d}"
    return f"CTR{year}{day_of_year}{batch_seq}"


def ask_blip2(image: Image.Image, question: str) -> str:
    """Ask BLIP-2 a question about the image."""
    global blip_processor, blip_model, device
    
    prompt = f"Question: {question} Answer:"
    inputs = blip_processor(image, prompt, return_tensors="pt").to(device, torch.float16 if device == "cuda" else torch.float32)
    
    outputs = blip_model.generate(
        **inputs,
        max_new_tokens=50,
        num_beams=5,
        early_stopping=True
    )
    
    answer = blip_processor.decode(outputs[0], skip_special_tokens=True).strip()
    
    # Clean up the answer
    if "Answer:" in answer:
        answer = answer.split("Answer:")[-1].strip()
    if "Question:" in answer:
        answer = answer.split("Question:")[0].strip()
    
    return answer


def analyze_product_with_blip2(image: Image.Image, product_line: str) -> dict:
    """
    Analyze product image using BLIP-2 with manufacturing QA-specific questions.
    Returns structured analysis dict.
    """
    logger.info("Analyzing product image with BLIP-2 VQA...")
    
    line_info = PRODUCT_LINES.get(product_line, PRODUCT_LINES["skincare_serum"])
    
    # Base questions for all products
    questions = {
        "product_type": "What type of cosmetic product is shown in this image? Is it a bottle, jar, tube, lipstick, or container?",
        "description": "Describe the cosmetic product you see in this image in detail.",
        "color": "What color is the product or substance? Describe any color variations or inconsistencies.",
        "surface_condition": "Describe the surface condition of the product. Is it smooth, textured, or does it show any defects?",
        "contamination": "Do you see any foreign particles, specks, debris, or contamination in or on the product?",
        "packaging_condition": "What is the condition of the packaging? Are there any cracks, dents, scratches, or damage?",
        "label_condition": "If there is a label visible, is it properly aligned and free of defects?",
        "cap_seal": "If visible, does the cap or seal appear properly secured and intact?",
        "overall_condition": "Is this product in perfect condition or does it have any defects, damage, or problems?"
    }
    
    # Add product-line specific questions
    if product_line == "lip_products":
        questions.update({
            # Very direct questions about damage - answer should reveal problems
            "is_broken": "Is this lipstick broken? Answer yes or no and explain what you see.",
            "tip_damage": "Look at the top of the lipstick bullet. Is it broken off, snapped, chipped, cracked, or damaged in any way?",
            "structural_integrity": "Is the lipstick bullet whole and complete, or is part of it broken off, missing, or separated?",
            "shape_normal": "Does this lipstick have a normal bullet shape, or is it deformed, bent, tilted, crooked, or misshapen?",
            "defect_visible": "Do you see any defects, damage, breaks, cracks, or problems with this lipstick? Describe any issues.",
            "quality_assessment": "Would this lipstick pass quality control, or does it have visible damage that would make it unsellable?"
        })
    elif product_line == "skincare_serum":
        questions.update({
            "fill_level": "Looking at the bottle, does it appear to be full, partially filled, or underfilled?",
            "liquid_clarity": "Is the liquid clear or does it appear cloudy, separated, or contain particles?",
            "dropper_condition": "If visible, is the dropper intact and functioning properly?"
        })
    elif product_line == "moisturizer":
        questions.update({
            "fill_level": "Looking at the container, does it appear to be full, partially filled, or underfilled?",
            "texture_uniformity": "Does the cream or lotion appear smooth and uniform, or is it lumpy, separated, or inconsistent?",
            "surface_skin": "Is there any discoloration, drying, or film on the surface of the cream?"
        })
    elif product_line == "mascara":
        questions.update({
            "wand_condition": "If the wand is visible, is it intact and properly formed?",
            "tube_integrity": "Is the mascara tube intact without cracks or leaks?",
            "formula_condition": "Does the mascara formula appear normal, or is it dried, clumpy, or abnormal?"
        })
    else:
        questions.update({
            "fill_level": "Looking at the container, does it appear to be full, partially filled, or underfilled?"
        })
    
    results = {}
    for key, question in questions.items():
        answer = ask_blip2(image, question)
        logger.info(f"{key}: {answer}")
        results[key] = answer
    
    return results


def generate_text_for_field(field_name: str, context: str, severity: str = "PASS", max_tokens: int = 50) -> str:
    """Use LLM to generate text for a specific field given context."""
    global text_generator
    
    if text_generator is None:
        return ""
    
    # Manufacturing/factory-focused prompts
    prompts = {
        "defect_description": f"<|user|>In one sentence, describe this manufacturing defect for a QA report: {context}</s><|assistant|>",
        "corrective_action_critical": f"<|user|>You are a manufacturing QA supervisor. In one sentence, state the immediate factory floor action for this critical defect: {context}. Focus on stopping production, quarantine, and escalation.</s><|assistant|>",
        "corrective_action_major": f"<|user|>You are a manufacturing QA supervisor. In one sentence, state the factory floor corrective action for this major defect: {context}. Focus on quarantine, documentation, and line review.</s><|assistant|>",
        "corrective_action_minor": f"<|user|>You are a manufacturing QA supervisor. In one sentence, state the documentation action for this minor defect: {context}. Focus on logging and monitoring.</s><|assistant|>",
        "root_cause": f"<|user|>You are a manufacturing engineer. In one sentence, suggest a likely production root cause for: {context}. Consider equipment, materials, or process issues.</s><|assistant|>"
    }
    
    if field_name == "corrective_action":
        prompt_key = f"corrective_action_{severity.lower()}"
        prompt = prompts.get(prompt_key, prompts["corrective_action_minor"])
    else:
        prompt = prompts.get(field_name, f"<|user|>Briefly describe: {context}</s><|assistant|>")
    
    try:
        result = text_generator(
            prompt,
            max_new_tokens=max_tokens,
            temperature=0.5,
            do_sample=True,
            pad_token_id=text_generator.tokenizer.eos_token_id
        )
        generated = result[0]['generated_text']
        if "<|assistant|>" in generated:
            response = generated.split("<|assistant|}") [-1].strip()
            # Also try the other delimiter in case
            if "<|assistant|>" in response:
                response = response.split("<|assistant|>")[-1].strip()
            for prefix in ["Sure!", "Sure,", "Of course!", "Certainly!", "1.", "1)", "Action:"]:
                if response.startswith(prefix):
                    response = response[len(prefix):].strip()
            if ". " in response:
                response = response.split(". ")[0] + "."
            return response
        return ""
    except Exception as e:
        logger.warning(f"Text generation failed for {field_name}: {e}")
        return ""


def generate_llm_detail(defect_type: str, severity: str, max_tokens: int = 25) -> str:
    """
    Generate a short LLM detail snippet for the hybrid template system.
    Returns a brief, specific phrase to insert into templates.
    """
    global text_generator
    
    if text_generator is None:
        return ""
    
    # Very focused prompts that ask for just a short action phrase
    if severity == "CRITICAL":
        prompt = f"<|user|>Complete this sentence with 5-10 words about factory action for {defect_type}: 'Immediately'</s><|assistant|>"
    elif severity == "MAJOR":
        prompt = f"<|user|>Complete this sentence with 5-10 words about documenting {defect_type}: 'Photograph the defect and'</s><|assistant|>"
    else:
        prompt = f"<|user|>Complete this sentence with 5-10 words about logging {defect_type}: 'Record the'</s><|assistant|>"
    
    try:
        result = text_generator(
            prompt,
            max_new_tokens=max_tokens,
            temperature=0.4,  # Lower temperature for more focused output
            do_sample=True,
            pad_token_id=text_generator.tokenizer.eos_token_id
        )
        generated = result[0]['generated_text']
        
        if "<|assistant|>" in generated:
            response = generated.split("<|assistant|}") [-1].strip()
            if "<|assistant|>" in response:
                response = response.split("<|assistant|>")[-1].strip()
            
            # Clean up common prefixes
            for prefix in ["Sure!", "Sure,", "Of course!", "Certainly!", "Immediately", "Photograph the defect and", "Record the"]:
                if response.lower().startswith(prefix.lower()):
                    response = response[len(prefix):].strip()
            
            # Take just the first sentence/phrase, limit length
            if ". " in response:
                response = response.split(". ")[0]
            if "," in response and len(response.split(",")[0]) > 10:
                response = response.split(",")[0]
            
            # Validate it's reasonable content
            bad_phrases = ["your", "you", "customer", "consumer", "application", "apply",
               "here's", "here is", "revised version", "sentence with", "added words",
               "descriptive words", "complete the", "completing"]
            if any(phrase in response.lower() for phrase in bad_phrases):
                return ""
            
            # Capitalize first letter and ensure it ends properly
            if response and len(response) > 5:
                response = response.strip().rstrip('.')
                return response
        
        return ""
    except Exception as e:
        logger.warning(f"LLM detail generation failed: {e}")
        return ""


def build_hybrid_corrective_action(severity: str, defect_type: str, batch_id: str) -> str:
    """
    Build corrective action using hybrid template + LLM system.
    """
    if severity not in CORRECTIVE_ACTION_TEMPLATES:
        return "Document and review per standard QA procedures."
    
    template_data = CORRECTIVE_ACTION_TEMPLATES[severity]
    template = template_data["template"]
    fallback = template_data["fallback_detail"]
    
    # Try to get LLM-generated detail
    llm_detail = generate_llm_detail(defect_type, severity)
    
    # Use fallback if LLM fails or returns empty/short
    if not llm_detail or len(llm_detail.strip()) < 5:
        llm_detail = fallback
    else:
        # Ensure proper capitalization and punctuation
        llm_detail = llm_detail[0].upper() + llm_detail[1:] if llm_detail else fallback
        if not llm_detail.endswith('.'):
            llm_detail += '.'
    
    # Build final corrective action from template
    corrective_action = template.format(
        batch_id=batch_id,
        llm_detail=llm_detail
    )
    
    return corrective_action


def build_hybrid_root_cause(defect_type: str) -> str:
    """
    Build root cause analysis using hybrid template + LLM system.
    """
    # Determine which template to use based on defect type
    if "broken" in defect_type or "damage" in defect_type or "snap" in defect_type:
        template_key = "broken"
    elif "contamin" in defect_type or "particle" in defect_type or "foreign" in defect_type:
        template_key = "contamination"
    elif "color" in defect_type or "discolor" in defect_type:
        template_key = "color"
    elif "fill" in defect_type:
        template_key = "fill"
    elif "deform" in defect_type or "misshapen" in defect_type or "bent" in defect_type:
        template_key = "deformed"
    else:
        template_key = "default"
    
    template_data = ROOT_CAUSE_TEMPLATES[template_key]
    template = template_data["template"]
    fallback = template_data["fallback_detail"]
    
    # Try to get LLM-generated detail
    llm_detail = generate_llm_detail(defect_type, "MAJOR")  # Use MAJOR-style prompt for root cause
    
    # Use fallback if LLM fails
    if not llm_detail or len(llm_detail.strip()) < 5:
        llm_detail = fallback
    else:
        llm_detail = llm_detail[0].upper() + llm_detail[1:] if llm_detail else fallback
        if not llm_detail.endswith('.'):
            llm_detail += '.'
    
    # Build final root cause from template
    root_cause = template.format(llm_detail=llm_detail)
    
    return root_cause


def determine_severity_and_defects(image_analysis: dict, product_line: str) -> tuple:
    """
    Analyze VLM output to determine severity level and defect list.
    Returns (severity, defects_list, defect_details).
    """
    all_text = " ".join(str(v) for v in image_analysis.values()).lower()
    
    defects = []
    defect_details = {}
    
    # Critical defect indicators - contamination and safety issues
    critical_keywords = [
        "mold", "foreign object", "contamination", "contaminated", "debris", "particle", "speck", 
        "insect", "hair", "fiber", "wrong color", "completely different", "wrong product",
        "bacteria", "fungus", "spoiled", "rotten", "separated severely", "foreign material",
        "metal", "glass shard", "unknown substance", "biological", "growth"
    ]
    
    # Major defect indicators - structural damage and significant quality issues
    major_keywords = [
        # Structural/physical damage - core terms
        "broken", "snapped", "cracked", "shattered", "split", "fractured", "chipped",
        "bent", "tilted", "leaning", "crooked", "deformed", "warped", "twisted",
        "falling", "detached", "separated", "loose", "unstable", "collapsed",
        "damaged", "destroyed", "ruined", "defective", "malformed",
        # Lipstick specific damage terms
        "broken off", "snapped off", "tip broken", "tip damaged", "bullet damaged",
        "not intact", "not straight", "falling off", "come off", "came off",
        "not whole", "missing piece", "missing part", "partial", "incomplete",
        "severed", "disconnected", "top broken", "head broken",
        # Yes/No damage confirmations (from direct questions)
        "yes, broken", "yes broken", "yes it is broken", "is broken", 
        "yes, damaged", "yes damaged", "yes it is damaged",
        "not pass", "would not pass", "fail quality", "unsellable", "not sellable",
        "has damage", "has defects", "has problems", "has issues",
        # Color/appearance issues
        "discolored", "discoloration", "dark spot", "yellowing", "browning", "oxidized",
        "wrong shade", "color mismatch", "faded", "bleached",
        # Texture/consistency issues  
        "curdled", "lumpy", "grainy", "crystallized", "dried out",
        "congealed", "solidified", "melted", "liquified",
        # Fill/quantity issues
        "underfilled", "overfilled", "empty", "half full", "missing product",
        # Seal/closure issues
        "leaked", "leaking", "spilled", "damaged seal", "broken seal", "open", "unsealed",
        "not sealed", "improper seal", "compromised",
        # Packaging issues
        "dented", "crushed", "punctured", "torn", "ripped"
    ]
    
    # Minor defect indicators - cosmetic issues that don't affect product integrity
    minor_keywords = [
        "slight variation", "minor", "small scratch", "tiny", "barely visible",
        "slightly off", "bit different", "little", "subtle", "faint",
        "cosmetic scratch", "scuff", "minor misalignment", "slight discoloration",
        "minor imperfection", "small mark", "light scratch", "surface mark",
        "minor variation", "slight difference"
    ]
    
    # Product-line specific detection
    if product_line == "lip_products":
        # Additional checks for lipstick structural issues
        lip_damage_indicators = [
            "broken", "snapped", "bent", "tilted", "leaning", "crooked", 
            "deformed", "damaged", "chipped", "not intact", "falling",
            "separated", "detached", "off center", "misaligned", "twisted",
            "not straight", "angled", "slanted", "tipped", "loose",
            "yes", "defect", "issue", "problem", "fail", "unsellable"
        ]
        
        # Check specific lipstick analysis fields for damage indicators
        lip_fields = ["is_broken", "tip_damage", "structural_integrity", 
                      "shape_normal", "defect_visible", "quality_assessment"]
        
        for field in lip_fields:
            if field in image_analysis:
                field_text = str(image_analysis[field]).lower()
                # Check for affirmative damage responses
                if any(indicator in field_text for indicator in lip_damage_indicators):
                    # Make sure it's actually indicating damage, not denying it
                    negation_phrases = ["no damage", "not broken", "no defect", "no issue", 
                                       "no problem", "intact", "perfect", "normal shape",
                                       "would pass", "good condition", "no visible"]
                    if not any(neg in field_text for neg in negation_phrases):
                        defect_msg = f"MAJOR: Lipstick damage detected - {field}: {image_analysis[field][:50]}"
                        if defect_msg not in defects:
                            defects.append(defect_msg)
                            defect_details["structural_damage"] = image_analysis.get(field, "Structural damage detected")
        
        # Also check general fields for lip damage
        for indicator in lip_damage_indicators[:20]:  # Skip yes/defect/etc for general check
            if indicator in all_text:
                # Verify it's not in a negation context
                negation_check = all_text.replace(indicator, f"[{indicator}]")
                if f"not [{indicator}]" not in negation_check and f"no [{indicator}]" not in negation_check:
                    defect_msg = f"MAJOR: Lipstick structural issue - {indicator}"
                    if defect_msg not in defects and not any(indicator in d for d in defects):
                        defects.append(defect_msg)
                        defect_details["structural_damage"] = image_analysis.get("description", "Structural damage detected")
    
    # Check for critical defects
    for keyword in critical_keywords:
        if keyword in all_text:
            defects.append(f"CRITICAL: {keyword.title()} detected")
            defect_details["contamination"] = image_analysis.get("contamination", "Contamination detected")
    
    # Check overall_condition and quality_assessment for explicit defect indicators
    overall_condition = str(image_analysis.get("overall_condition", "")).lower()
    quality_assessment = str(image_analysis.get("quality_assessment", "")).lower()
    
    # If overall condition mentions defects/damage/problems
    overall_defect_terms = ["defect", "damage", "problem", "issue", "broken", "crack", "not perfect", 
                           "imperfect", "flaw", "fault", "not in perfect", "has some"]
    for term in overall_defect_terms:
        if term in overall_condition or term in quality_assessment:
            defect_msg = f"MAJOR: Overall condition indicates {term}"
            if defect_msg not in defects and not any(term in d.lower() for d in defects):
                defects.append(defect_msg)
                defect_details["overall_condition"] = image_analysis.get("overall_condition", "Condition issue detected")
    
    # Check for major defects
    for keyword in major_keywords:
        if keyword in all_text:
            # Avoid double-counting if already added as lip damage
            defect_msg = f"MAJOR: {keyword.title()} observed"
            if defect_msg not in defects and f"MAJOR: Lipstick structural damage - {keyword}" not in defects:
                defects.append(defect_msg)
                if "color" in keyword or "discolor" in keyword:
                    defect_details["color_issue"] = image_analysis.get("color", "Color variation detected")
                if "fill" in keyword or "empty" in keyword:
                    defect_details["fill_issue"] = image_analysis.get("fill_level", "Fill level issue")
                if "seal" in keyword or "crack" in keyword or "leak" in keyword:
                    defect_details["packaging_issue"] = image_analysis.get("packaging_condition", "Packaging issue")
                if any(w in keyword for w in ["broken", "snap", "bent", "tilt", "damage", "deform"]):
                    defect_details["structural_issue"] = image_analysis.get("surface_condition", "Structural issue detected")
    
    # Check for minor defects
    for keyword in minor_keywords:
        if keyword in all_text:
            defects.append(f"MINOR: {keyword.title()} noted")
    
    # Determine overall severity
    if any("CRITICAL" in d for d in defects):
        severity = "CRITICAL"
    elif any("MAJOR" in d for d in defects):
        severity = "MAJOR"
    elif any("MINOR" in d for d in defects):
        severity = "MINOR"
    else:
        severity = "PASS"
        defects = ["No defects detected - Product within specification"]
    
    return severity, defects, defect_details


def build_qa_report(image_analysis: dict, product_line: str, custom_instructions: str = "") -> str:
    """
    Create QA inspection report using structured template with LLM-generated content.
    """
    line_info = PRODUCT_LINES.get(product_line, PRODUCT_LINES["skincare_serum"])
    
    product_type = image_analysis.get("product_type", "Cosmetic product")
    description = image_analysis.get("description", "No description available")
    color = image_analysis.get("color", "Unable to determine")
    fill_level = image_analysis.get("fill_level", "Unable to determine")
    surface = image_analysis.get("surface_condition", "Unable to determine")
    contamination = image_analysis.get("contamination", "None detected")
    packaging = image_analysis.get("packaging_condition", "Unable to determine")
    label = image_analysis.get("label_condition", "Unable to determine")
    cap_seal = image_analysis.get("cap_seal", "Unable to determine")
    
    # Determine severity and defects
    severity, defects, defect_details = determine_severity_and_defects(image_analysis, product_line)
    severity_info = SEVERITY_LEVELS[severity]
    
    # Generate batch ID for this inspection
    batch_id = generate_batch_id()
    
    # Generate corrective action and root cause using hybrid template + LLM system
    if severity != "PASS":
        # Extract the core issue type from the first defect
        first_defect = defects[0] if defects else "Unknown defect"
        
        # Simplify the defect type for template selection
        if "tip_damage" in first_defect or "tip" in first_defect.lower():
            simple_issue = "broken or damaged lipstick tip"
        elif "structural" in first_defect.lower():
            simple_issue = "structural damage to product"
        elif "shape" in first_defect.lower() or "crooked" in first_defect.lower() or "bent" in first_defect.lower():
            simple_issue = "deformed or misshapen product"
        elif "quality" in first_defect.lower() or "unsellable" in first_defect.lower():
            simple_issue = "product failed quality inspection"
        elif "broken" in first_defect.lower():
            simple_issue = "broken product"
        elif "contamination" in first_defect.lower() or "particle" in first_defect.lower():
            simple_issue = "contamination detected"
        elif "discolor" in first_defect.lower() or "color" in first_defect.lower():
            simple_issue = "color defect or discoloration"
        elif "fill" in first_defect.lower():
            simple_issue = "incorrect fill level"
        else:
            simple_issue = "product defect"
        
        # Use hybrid template + LLM system
        corrective_action = build_hybrid_corrective_action(severity, simple_issue, batch_id)
        root_cause = build_hybrid_root_cause(simple_issue)
    else:
        corrective_action = "None required - Product approved for release to packaging and distribution."
        root_cause = "N/A - No defects detected. Product meets all quality specifications."
    
    # Clean up and deduplicate defects list
    # Remove duplicates and overly verbose entries
    seen_defects = set()
    cleaned_defects = []
    for defect in defects:
        # Create a simplified key for deduplication
        defect_lower = defect.lower()
        
        # Skip if we've seen a similar defect
        skip = False
        for seen in seen_defects:
            # Check for substantial overlap
            if seen in defect_lower or defect_lower in seen:
                skip = True
                break
        
        if not skip:
            # Clean up the defect text - remove field names and truncation
            clean_defect = defect
            if " - tip_damage:" in defect or " - structural_integrity:" in defect or " - shape_normal:" in defect or " - quality_assessment:" in defect:
                # Simplify field-specific detections
                if "tip_damage" in defect:
                    clean_defect = "MAJOR: Lipstick tip/bullet damage detected"
                elif "structural_integrity" in defect:
                    clean_defect = "MAJOR: Structural integrity compromised"
                elif "shape_normal" in defect:
                    clean_defect = "MAJOR: Abnormal shape/deformation detected"
                elif "quality_assessment" in defect:
                    clean_defect = "MAJOR: Failed quality assessment - unsellable"
            
            cleaned_defects.append(clean_defect)
            seen_defects.add(defect_lower[:30])  # Use first 30 chars as key
    
    # Limit to top 5 most important findings to avoid clutter
    if len(cleaned_defects) > 5:
        cleaned_defects = cleaned_defects[:5]
        cleaned_defects.append("... and additional defects noted")
    
    defects_formatted = "\n   • ".join(cleaned_defects)
    
    # Convert defects list to a searchable string for checklist logic
    defects_text = " ".join(cleaned_defects).lower()
    
    # Build inspection checklist results - NOW DRIVEN BY SECTION 4 DEFECT DETECTION
    inspection_results = []
    inspection_points = line_info["inspection_points"]
    
    for point in inspection_points:
        point_lower = point.lower()
        
        # Determine status based on whether related defects were found in Section 4
        if "fill" in point_lower:
            has_defect = any(w in defects_text for w in ["fill", "underfill", "overfill", "empty"])
            status = "✗ FAIL" if has_defect and severity in ["CRITICAL", "MAJOR"] else "⚠️ REVIEW" if has_defect else "✓ PASS"
        elif "color" in point_lower:
            has_defect = any(w in defects_text for w in ["color", "discolor", "shade", "faded"])
            status = "✗ FAIL" if has_defect and severity in ["CRITICAL", "MAJOR"] else "⚠️ REVIEW" if has_defect else "✓ PASS"
        elif "contamination" in point_lower or "particulate" in point_lower:
            has_defect = any(w in defects_text for w in ["contamin", "particle", "debris", "foreign", "speck"])
            status = "✗ FAIL" if has_defect else "✓ PASS"
        elif "seal" in point_lower or "cap" in point_lower:
            has_defect = any(w in defects_text for w in ["seal", "leak", "open", "unsealed"])
            status = "✗ FAIL" if has_defect and severity in ["CRITICAL", "MAJOR"] else "⚠️ REVIEW" if has_defect else "✓ PASS"
        elif "label" in point_lower:
            has_defect = any(w in defects_text for w in ["label", "misalign"])
            status = "⚠️ REVIEW" if has_defect else "✓ PASS"
        elif "tip" in point_lower or "integrity" in point_lower:
            # Lipstick tip/integrity - check for structural defects
            has_defect = any(w in defects_text for w in ["tip", "broken", "snapped", "structural", "integrity", "damage"])
            status = "✗ FAIL" if has_defect else "✓ PASS"
        elif "surface" in point_lower or "finish" in point_lower:
            has_defect = any(w in defects_text for w in ["surface", "crack", "chip", "gouge", "rough"])
            status = "✗ FAIL" if has_defect and severity in ["CRITICAL", "MAJOR"] else "⚠️ REVIEW" if has_defect else "✓ PASS"
        elif "component" in point_lower or "assembly" in point_lower:
            has_defect = any(w in defects_text for w in ["broken", "damaged", "structural", "deform", "shape"])
            status = "✗ FAIL" if has_defect else "✓ PASS"
        elif "texture" in point_lower or "uniformity" in point_lower:
            has_defect = any(w in defects_text for w in ["texture", "lumpy", "separated", "grainy"])
            status = "✗ FAIL" if has_defect and severity in ["CRITICAL", "MAJOR"] else "⚠️ REVIEW" if has_defect else "✓ PASS"
        else:
            # Default: pass unless we have major/critical severity
            status = "⚠️ REVIEW" if severity in ["CRITICAL", "MAJOR"] else "✓ PASS"
        
        inspection_results.append(f"{point}: {status}")
    
    inspection_checklist = "\n   ".join(inspection_results)
    
     
    # Build the structured report
    report = f"""1. PRODUCT IDENTIFICATION
   Product Line: {line_info['name']}
   Container Type: {line_info['container_type']}
   Visual ID: {product_type.capitalize()}
   Description: {description.capitalize()}

2. VISUAL INSPECTION RESULTS
   {inspection_checklist}

3. DEFECT SUMMARY
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   SEVERITY LEVEL: {severity}
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Required Action: {severity_info['action']}
   Response Time: {severity_info['response_time']}
   
   Findings:
   • {defects_formatted}

4. CORRECTIVE ACTION
   Recommendation: {corrective_action}
   Probable Root Cause: {root_cause}

5. QUALITY DISPOSITION
   Status: {'REJECT - Do not release' if severity in ['CRITICAL', 'MAJOR'] else 'HOLD - Pending review' if severity == 'MINOR' else 'APPROVED - Release to packaging'}
   Confidence: HIGH (BLIP-2 multi-point visual analysis){f'''
   
6. ADDITIONAL NOTES
   {custom_instructions}''' if custom_instructions else ""}"""
    
    return report, severity


def send_alert_email(report_data: dict) -> bool:
    """Send email alert for critical/major defects."""
    if not SMTP_HOST or not SMTP_USER:
        logger.info("Email not configured - alert would be sent to production lead")
        return False
    
    try:
        msg = MIMEMultipart()
        msg['From'] = SMTP_USER
        msg['To'] = ALERT_EMAIL
        msg['Subject'] = f"⚠️ QA ALERT: {report_data['severity']} Defect Detected - {report_data['inspection_id']}"
        
        body = f"""
COUNTER MANUFACTURING QA ALERT
==============================

Inspection ID: {report_data['inspection_id']}
Batch ID: {report_data['batch_id']}
Timestamp: {report_data['timestamp']}
Product Line: {report_data['product_line']}

SEVERITY: {report_data['severity']}
ACTION REQUIRED: {SEVERITY_LEVELS[report_data['severity']]['action']}

Summary:
{report_data['summary']}

Please review immediately and take appropriate action.

---
This is an automated alert from the Counter QA Vision System
Powered by HP ZGX Nano AI Station
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
        
        logger.info(f"Alert email sent to {ALERT_EMAIL}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to send alert email: {e}")
        return False


def analyze_product(image: Image.Image, product_line: str, custom_instructions: str = "") -> dict:
    """Analyze a product image and generate a QA inspection report."""
    global blip_model
    
    if blip_model is None:
        load_models()
    
    if blip_model is None:
        raise RuntimeError("BLIP-2 model not available")
    
    # Ensure RGB
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Resize if too large
    max_size = 1024
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        image = image.resize(new_size, Image.LANCZOS)
    
    # Get structured analysis from BLIP-2
    image_analysis = analyze_product_with_blip2(image, product_line)
    
    # Generate QA report
    analysis_report, severity = build_qa_report(image_analysis, product_line, custom_instructions)
    
    # Generate IDs
    inspection_id = generate_inspection_id()
    batch_id = generate_batch_id()
    timestamp = datetime.now(timezone.utc).strftime("%d %b %Y %H:%M:%S UTC").upper()
    
    line_info = PRODUCT_LINES.get(product_line, PRODUCT_LINES["skincare_serum"])
    severity_info = SEVERITY_LEVELS[severity]
    
    # Build response
    report = {
        "inspection_id": inspection_id,
        "batch_id": batch_id,
        "timestamp": timestamp,
        "product_line": line_info["name"],
        "product_line_id": product_line,
        "severity": severity,
        "severity_color": severity_info["color"],
        "action_required": severity_info["action"],
        "response_time": severity_info["response_time"],
        "analysis": analysis_report,
        "raw_analysis": image_analysis,
        "alert_sent": False,
        "alert_recipient": ALERT_EMAIL,
        "generated_at": datetime.now(timezone.utc).isoformat()
    }
    
    # Send alert for critical/major defects
    if severity in ["CRITICAL", "MAJOR"]:
        defects = [d for d in analysis_report.split("•") if severity in d.upper()]
        summary = defects[0] if defects else f"{severity} defect detected"
        
        alert_data = {
            "inspection_id": inspection_id,
            "batch_id": batch_id,
            "timestamp": timestamp,
            "product_line": line_info["name"],
            "severity": severity,
            "summary": summary
        }
        report["alert_sent"] = send_alert_email(alert_data)
        report["alert_message"] = f"Alert triggered for {severity} severity - Production Lead notified"
    
    return report


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main application page."""
    possible_paths = [
        Path(__file__).parent.parent / "frontend" / "index.html",
        Path(__file__).parent / "frontend" / "index.html",
        Path(__file__).parent / "index.html",
    ]
    
    for html_path in possible_paths:
        if html_path.exists():
            return HTMLResponse(content=html_path.read_text())
    
    return HTMLResponse(content="<h1>Counter Manufacturing QA System</h1><p>index.html not found</p>")


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "blip2_model_loaded": blip_model is not None,
        "text_generator_loaded": text_generator is not None,
        "device": device,
        "email_configured": bool(SMTP_HOST),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.get("/api/product-lines")
async def get_product_lines():
    """Get available product lines for inspection."""
    return {
        "product_lines": [
            {
                "id": k, 
                "name": v["name"], 
                "products": v["products"],
                "container_type": v["container_type"]
            }
            for k, v in PRODUCT_LINES.items()
        ]
    }


@app.get("/api/severity-levels")
async def get_severity_levels():
    """Get severity level definitions."""
    return {"severity_levels": SEVERITY_LEVELS}


@app.post("/api/inspect")
async def inspect_endpoint(
    image: UploadFile = File(...),
    product_line: str = Form("skincare_serum"),
    custom_instructions: str = Form("")
):
    """Analyze an uploaded product image and generate a QA inspection report."""
    
    # Validate file type
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Read image data
    image_data = await image.read()
    
    if len(image_data) > 20 * 1024 * 1024:  # 20MB limit
        raise HTTPException(status_code=400, detail="Image too large (max 20MB)")
    
    # Open image
    try:
        img = Image.open(io.BytesIO(image_data))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")
    
    # Perform inspection
    try:
        report = analyze_product(img, product_line, custom_instructions)
        return JSONResponse(content=report)
    except Exception as e:
        logger.error(f"Inspection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Inspection failed: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """Pre-load the models on startup."""
    print("=" * 60)
    print("Counter Manufacturing Quality Assurance System")
    print("Powered by HP ZGX Nano AI Station")
    print("Using BLIP-2 FLAN-T5-XL for Visual Product Inspection")
    print("=" * 60)
    try:
        load_models()
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        print(f"Warning: {e}")
        print("Models will be loaded on first request.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)