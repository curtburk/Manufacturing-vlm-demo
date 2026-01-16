# Counter Manufacturing Quality Assurance System

**Powered by HP ZGX Nano AI Station**

A Vision Language Model (VLM) demonstration for cosmetics manufacturing defect detection with severity classification and event-driven alert generation.

---

## Overview

This demo showcases on-premises AI capability for cosmetics manufacturing quality control. The system:

- Analyzes product images using **BLIP-2 FLAN-T5-XL** vision-language model
- Detects defects including discoloration, contamination, structural damage, and packaging problems
- Classifies defect severity (CRITICAL/MAJOR/MINOR/PASS)
- Generates structured QA inspection reports with factory-floor corrective actions
- Uses a **hybrid template + LLM system** for reliable, professional recommendations
- Triggers event-driven email alerts for critical and major defects
- Runs entirely on-premises with no cloud dependency

### Key Features

| Feature | Description |
|---------|-------------|
| **Visual Inspection** | BLIP-2 answers targeted questions about product condition, structural integrity, contamination, and quality |
| **Severity Classification** | Automatic severity level (CRITICAL/MAJOR/MINOR/PASS) based on comprehensive defect analysis |
| **Hybrid Template + LLM** | Reliable corrective actions using structured templates with LLM-generated details |
| **Event-Driven Alerts** | Email notifications to production line lead for critical/major defects |
| **Unified Detection Logic** | Inspection checklist and defect summary driven by single detection engine for consistency |

### Value Propositions

| Benefit | Description |
|---------|-------------|
| **Consistent Quality** | AI-powered inspection eliminates human fatigue and subjectivity |
| **Real-Time Detection** | Immediate defect identification on the production line |
| **Factory-Ready Output** | Corrective actions written for production floor workers, not consumers |
| **Traceability** | Full audit trail with inspection IDs, batch numbers, and timestamps |
| **Compliance Ready** | Structured reports support regulatory compliance requirements |

---

## Architecture

### Models Used

| Model | Purpose | Size |
|-------|---------|------|
| **Salesforce/blip2-flan-t5-xl** | Visual Question Answering - analyzes product appearance, defects, structural integrity | ~15GB |
| **TinyLlama/TinyLlama-1.1B-Chat-v1.0** | Generates detail snippets for hybrid template system | ~2GB |

### Hybrid Template + LLM System

The system uses a hybrid approach for generating corrective actions and root cause analysis:

1. **Structured Templates** define the overall format and required actions per severity level
2. **TinyLlama** generates short, specific detail snippets (5-10 words) to customize each template
3. **Fallback System** ensures professional output even if LLM generation fails

**Example Output:**
```
Recommendation: Remove defective unit from line and place in quarantine hold area 
with red tag. [LLM: Document the broken tip with close-up photographs.] Notify 
Line Supervisor within 1 hour. Review last 50 units for similar defects.
```

### Technology Stack

- **Backend**: FastAPI (Python)
- **Frontend**: HTML/CSS/JavaScript with clean beauty-themed UI
- **ML Framework**: Hugging Face Transformers
- **Inference**: PyTorch with CUDA support
- **Alerting**: SMTP email integration (optional)

---

## Quick Start

### Prerequisites

- HP ZGX Nano AI Station (or NVIDIA GPU with 24GB+ VRAM)
- Python 3.10+
- CUDA 12.0+

### Installation

```bash
# Navigate to demo directory
cd /path/to/counter-qa-demo

# Run installation script
chmod +x install.sh
./install.sh

# Start the demo
./start_demo.sh

# Or for remote access
./start_demo_remote.sh
```

### Directory Structure

```
counter-qa-demo/
├── backend/
│   ├── main.py              # FastAPI application with hybrid template system
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── index.html           # Web interface
│   └── hp_logo.png          # HP logo (optional)
├── counter-env/             # Python virtual environment
├── install.sh               # Installation script
├── start_demo.sh            # Local startup script
├── start_demo_remote.sh     # Remote access startup script
└── README.md                # This file
```

### Example data images

There are two images included in this repo that can be used for demo purposes 

### First Run

On first startup, the models will be downloaded from Hugging Face (~17GB total). This may take 10-15 minutes depending on network speed. Subsequent starts will be much faster as models are cached.

---

## Product Lines

The system supports inspection for Counter's product categories:

| Product Line | Products | Key Inspection Points |
|--------------|----------|----------------------|
| **Skincare Serums** | All Bright C Serum, Countertime Tripeptide Radiance Serum | Fill level, color consistency, particulate matter |
| **Moisturizers & Creams** | Adaptive Moisture Lotion, Countermatch Recovery Sleeping Cream | Texture uniformity, color consistency, surface contamination |
| **Lip Products** | Sheer Lipstick, Beyond Gloss, Lip Conditioner | Tip integrity, structural damage, shape assessment, quality check |
| **Mascara & Eye Products** | Lengthening Mascara, Think Big All-In-One Mascara | Wand integrity, formula consistency, tube seal |
| **Body Care** | Body Butter, Citrus Mimosa Body Wash, Hand Cream | Fill level, texture uniformity, pump mechanism |
| **Sun Protection** | Countersun Mineral Sunscreen SPF 30, Dew Skin Tinted Moisturizer | Color consistency, separation check, UV filter distribution |

### Lip Products - Enhanced Detection

For lip products, the system asks targeted structural questions:
- "Is this lipstick broken? Answer yes or no and explain what you see."
- "Is the lipstick bullet whole and complete, or is part of it broken off, missing, or separated?"
- "Would this lipstick pass quality control, or does it have visible damage that would make it unsellable?"

---

## Severity Levels

| Level | Color | Action | Response Time | Examples |
|-------|-------|--------|---------------|----------|
| **CRITICAL** | 🔴 Red | STOP LINE - Immediate escalation | Immediate | Foreign object contamination, microbial growth, wrong product |
| **MAJOR** | 🟡 Orange | QUARANTINE - Hold for QA review | Within 1 hour | Broken/damaged product, significant discoloration, visible separation |
| **MINOR** | 🔵 Blue | FLAG - Document and monitor | End of shift | Slight color variation, minor label misalignment |
| **PASS** | 🟢 Green | RELEASE - Approved for distribution | N/A | All inspection points within specification |

---

## QA Inspection Report Format

Each inspection generates a structured report with 5 sections:

```
1. PRODUCT IDENTIFICATION
   Product Line: [Category]
   Container Type: [Type]
   Visual ID: [BLIP-2 product identification]
   Description: [BLIP-2 description]

2. VISUAL INSPECTION RESULTS
   [Checklist of inspection points with ✓ PASS / ⚠️ REVIEW / ✗ FAIL status]
   (Status driven by defect detection logic for consistency)

3. DEFECT SUMMARY
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   SEVERITY LEVEL: [CRITICAL/MAJOR/MINOR/PASS]
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Required Action: [Action based on severity]
   Response Time: [Required response time]
   
   Findings:
   • [List of detected defects - cleaned and deduplicated]

4. CORRECTIVE ACTION
   Recommendation: [Hybrid template + LLM factory-floor action]
   Probable Root Cause: [Hybrid template + LLM investigation guidance]

5. QUALITY DISPOSITION
   Status: [REJECT/HOLD/APPROVED]
   Confidence: HIGH (BLIP-2 multi-point visual analysis)
```

### Corrective Action Templates

**CRITICAL Severity:**
> IMMEDIATE: Stop production line and isolate batch [batch_id]. [LLM detail] Escalate to QA Manager and Plant Supervisor. Initiate root cause investigation per SOP-QA-001.

**MAJOR Severity:**
> Remove defective unit from line and place in quarantine hold area with red tag. [LLM detail] Notify Line Supervisor within 1 hour. Review last 50 units for similar defects.

**MINOR Severity:**
> Tag unit for QA review. [LLM detail] Log in production database and continue monitoring. Report at end-of-shift QA briefing.

---

## Email Alert System

### Configuration

Set environment variables before starting the demo:

```bash
export SMTP_HOST=smtp.your-server.com
export SMTP_PORT=587
export SMTP_USER=qa-alerts@counter.com
export SMTP_PASS=your-secure-password
export ALERT_EMAIL=production.lead@counter.com
```

### Alert Triggers

| Severity | Email Sent | Subject Format |
|----------|------------|----------------|
| CRITICAL | ✅ Yes | ⚠️ QA ALERT: CRITICAL Defect Detected - [Inspection ID] |
| MAJOR | ✅ Yes | ⚠️ QA ALERT: MAJOR Defect Detected - [Inspection ID] |
| MINOR | ❌ No | N/A (logged only) |
| PASS | ❌ No | N/A |

---

## Stakeholder Presentation Guide

### S.T.A.R. Framework

**Situation:**
Counter's clean beauty manufacturing requires rigorous quality control to maintain brand reputation and customer trust. Manual inspection is subjective, inconsistent, and labor-intensive. A single defective product reaching customers can damage years of brand building.

**Task:**
Implement automated visual inspection that:
- Detects defects with consistent accuracy
- Classifies severity for appropriate response
- Provides factory-floor actionable recommendations
- Alerts production leadership immediately for critical issues
- Maintains full traceability for compliance

**Action:**
Deploy HP ZGX Nano AI Station with on-premises VLM capability:
- BLIP-2 FLAN-T5-XL for multi-point visual analysis with targeted questions
- Hybrid template + LLM system for reliable corrective actions
- Unified detection logic ensuring report consistency
- Event-driven email alerts to production lead
- GPU-accelerated processing on NVIDIA Grace Blackwell

**Result:**
- Product inspection in 5-10 seconds vs. manual review (30-60 seconds)
- 100% consistency - no human fatigue or subjectivity
- Factory-ready recommendations - actionable by line workers
- Immediate alerts for critical defects - stop issues before they spread
- Full audit trail for FDA/regulatory compliance
- Estimated 80% reduction in inspection labor costs

### Demo Script

1. **Introduction** (1 min)
   - Show the clean, Counter-branded interface
   - Highlight "QUALITY CONTROL INSPECTION STATION" banner
   - Point out HP ZGX Nano branding and Vision AI status

2. **Upload Test Image** (1 min)
   - Use a cosmetic product image (serum bottle, cream jar, lipstick, etc.)
   - Select the appropriate product line
   - Click "Run Quality Inspection"

3. **Review Output** (2-3 min)
   - Walk through the inspection report sections
   - Highlight the severity level and color coding
   - Show the inspection checklist results (unified with defect detection)
   - Discuss the factory-floor corrective action recommendations

4. **Defect Demo** (2 min)
   - Upload an image with visible defects (e.g., broken lipstick)
   - Show how severity changes to MAJOR
   - Highlight the specific corrective actions and root cause analysis
   - Show the alert notification

5. **Value Discussion** (2 min)
   - Emphasize consistency and speed
   - Discuss brand protection value
   - Note regulatory compliance benefits
   - Address scalability and deployment options

### Suggested Test Images

For compelling demonstrations, use images showing:

**PASS scenarios:**
- Clean, properly filled serum bottles
- Well-formed lipstick tips with intact bullet shape
- Properly sealed cream jars
- Correct color consistency

**Defect scenarios:**
- Broken or snapped lipstick tips
- Bent, tilted, or deformed product
- Products with visible particles or contamination
- Discolored creams or serums
- Cracked or damaged packaging

---

## API Reference

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main application interface |
| `/api/health` | GET | Health check and model status |
| `/api/product-lines` | GET | List available product lines |
| `/api/severity-levels` | GET | Get severity level definitions |
| `/api/inspect` | POST | Analyze image and generate inspection report |

### POST /api/inspect

**Request:**
- `image`: Image file (multipart/form-data)
- `product_line`: Product line ID (default: "skincare_serum")
- `custom_instructions`: Additional inspection notes (optional)

**Response:**
```json
{
  "inspection_id": "QA-A-20260115-1234",
  "batch_id": "CTR2501501",
  "timestamp": "15 JAN 2026 14:30:00 UTC",
  "product_line": "Skincare Serums",
  "product_line_id": "skincare_serum",
  "severity": "PASS",
  "severity_color": "#22c55e",
  "action_required": "RELEASE - Approved for distribution",
  "response_time": "N/A",
  "analysis": "...[structured report text]...",
  "raw_analysis": {
    "product_type": "glass dropper bottle",
    "color": "clear amber liquid",
    "fill_level": "properly filled",
    ...
  },
  "alert_sent": false,
  "alert_recipient": "production.lead@counter.com",
  "generated_at": "2026-01-15T14:30:00Z"
}
```

---

## Troubleshooting

### Model Loading Errors

```bash
# Check CUDA availability
python3 -c "import torch; print(torch.cuda.is_available())"

# Check GPU memory
nvidia-smi

# Clear Hugging Face cache and re-download
rm -rf ~/.cache/huggingface/
python3 backend/main.py
```

### Out of Memory Errors

BLIP-2 FLAN-T5-XL requires ~15GB VRAM. If you encounter OOM errors:

```python
# In main.py, try loading with 8-bit quantization
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8bit=True)
blip_model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-flan-t5-xl",
    quantization_config=quantization_config
)
```

### Port Already in Use

```bash
# Find and kill existing process
lsof -i :8000
kill -9 <PID>
```

### Email Alerts Not Sending

```bash
# Verify SMTP configuration
echo $SMTP_HOST
echo $SMTP_USER

# Test SMTP connection
python3 -c "import smtplib; s = smtplib.SMTP('$SMTP_HOST', 587); s.starttls(); print('Connection OK')"
```

---

## Technical Specifications

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 16GB | 24GB+ |
| System RAM | 32GB | 64GB |
| Storage | 50GB | 100GB SSD |
| GPU | NVIDIA Ampere+ | NVIDIA Grace Blackwell (GB10) |

### Model Performance

| Metric | Value |
|--------|-------|
| Image Analysis Time | 5-10 seconds |
| Report Generation | 2-3 seconds |
| Total Processing | ~10-15 seconds |
| Throughput | ~4-6 inspections/minute |

---

## Integration Options

### Manufacturing Execution System (MES)

The `/api/inspect` endpoint can be called directly from MES systems to integrate visual inspection into production workflows.

### Conveyor Integration

For high-volume inspection, consider:
- Camera trigger integration via GPIO
- Batch processing mode
- Reject mechanism control via API webhook

### Data Export

All inspection data includes:
- Unique inspection ID for traceability
- Batch ID for lot tracking
- ISO 8601 timestamps
- JSON format for easy integration

---

## About Counter

Counter is a clean beauty company founded on the principle that beauty products should be safe, effective, and transparently made. With a "Never List" of over 1,800 restricted ingredients, Counter sets the standard for clean cosmetics.

This QA system helps ensure every Counter product meets the brand's exacting quality standards before reaching customers.

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Jan 2026 | Initial release with BLIP-2 visual inspection |
| 2.0 | Jan 2026 | Hybrid template + LLM system for corrective actions |
| 2.1 | Jan 2026 | Enhanced lip product detection with targeted VQA questions |
| 2.2 | Jan 2026 | Unified detection logic - inspection checklist driven by defect analysis |
| 2.3 | Jan 2026 | Removed redundant observations section, streamlined report format |

---

## Support

For technical assistance or demo customization, contact the HP ZGX Product Team.

**Powered by HP ZGX Nano AI Station**
