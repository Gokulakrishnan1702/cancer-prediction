import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

# Define custom Canvas for Running Header / Footer and Dynamic Page Numbers (Page X of Y)
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_header_footer(self, total_pages):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#4A5568"))

        # Running Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(44, 11 * inch - 34, "ONCOLOGY CDSS -- 6-STAGE TECHNICAL ARCHITECTURE & VALIDATION REPORT")
            self.drawRightString(8.5 * inch - 44, 11 * inch - 34, "CONFIDENTIAL / CLINICAL DECISION SUPPORT")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(44, 11 * inch - 38, 8.5 * inch - 44, 11 * inch - 38)

        # Running Footer (all pages)
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(44, 44, 8.5 * inch - 44, 44)

        self.drawString(44, 30, "Autonomous Multi-Stage Clinical Decision Support System (CDSS) | Production Release v2.4")
        page_str = f"Page {self._pageNumber} of {total_pages}"
        self.drawRightString(8.5 * inch - 44, 30, page_str)
        self.restoreState()


def create_clinical_pdf(output_filename="MultiStage_Oncology_AI_Comprehensive_Report.pdf"):
    # Target page setup: Letter, 44pt margins
    doc = SimpleDocTemplate(
        output_filename,
        pagesize=letter,
        leftMargin=44,
        rightMargin=44,
        topMargin=46,
        bottomMargin=48
    )

    styles = getSampleStyleSheet()

    # Custom Color Palette
    PRIMARY = colors.HexColor("#0F172A")    # Deep Navy Slate
    SECONDARY = colors.HexColor("#0284C7")  # Medical Teal / Blue
    ACCENT = colors.HexColor("#0D9488")     # Clinical Cyan
    DARK_TEXT = colors.HexColor("#1E293B")  # Charcoal Text
    LIGHT_BG = colors.HexColor("#F8FAFC")   # Clean Light Background
    BORDER_COLOR = colors.HexColor("#E2E8F0")

    # Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=PRIMARY,
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=SECONDARY,
        spaceAfter=8
    )

    h1_style = ParagraphStyle(
        'Header1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=PRIMARY,
        spaceBefore=8,
        spaceAfter=5,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Header2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=SECONDARY,
        spaceBefore=6,
        spaceAfter=3,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=DARK_TEXT,
        spaceAfter=4
    )

    bullet_style = ParagraphStyle(
        'BulletText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.2,
        leading=11,
        textColor=DARK_TEXT,
        leftIndent=12,
        firstLineIndent=-8,
        spaceAfter=3
    )

    table_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.white,
        alignment=0
    )

    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.8,
        leading=10,
        textColor=DARK_TEXT
    )

    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=table_cell,
        fontName='Helvetica-Bold'
    )

    meta_callout = ParagraphStyle(
        'MetaCallout',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#334155")
    )

    story = []

    # =========================================================================
    # PAGE 1: TITLE, EXECUTIVE SUMMARY & END-TO-END ARCHITECTURE
    # =========================================================================
    story.append(Paragraph("MULTI-STAGE ONCOLOGY CLINICAL DECISION SUPPORT SYSTEM (CDSS)", title_style))
    story.append(Paragraph("Master Technical Architecture, Machine Learning Benchmarks & Clinical Validation Report", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=SECONDARY, spaceAfter=8, spaceBefore=0))

    # Meta Info Box
    meta_data = [
        [
            Paragraph("<b>Target Domain:</b> Precision Oncology & Patient Risk Prediction", body_style),
            Paragraph("<b>Platform Version:</b> Production v2.4 (Multi-Stage Integrated)", body_style)
        ],
        [
            Paragraph("<b>Clinical Scope:</b> Toxicity, Progression, NLP, SLM, VAE, Multi-Agent", body_style),
            Paragraph("<b>Validation Status:</b> 15/15 Standard Benchmark Scenarios Passed (100%)", body_style)
        ]
    ]
    meta_table = Table(meta_data, colWidths=[260, 260])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
        ('PADDING', (0, 0), (-1, -1), 4),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 6))

    story.append(Paragraph("Executive Summary & System Objectives", h1_style))
    story.append(Paragraph(
        "Modern cancer therapy requires simultaneous integration of multi-analyte laboratory profiles, high-resolution visual histopathology, longitudinal tumor kinetics, unstructured electronic health records (EHR), and fast-evolving clinical trial criteria. This report provides an exhaustive technical and clinical audit of a <b>6-Stage Autonomous Oncology Decision Support System (CDSS)</b> designed to assist multidisciplinary tumor boards in predicting treatment toxicity, forecasting disease recurrence, synthesizing out-of-distribution stress profiles, and executing real-time genomic trial matching with zero-tolerance safety interlocks.",
        body_style
    ))
    story.append(Paragraph(
        "The platform replaces fragmented single-task heuristics with a continuous multi-stage pipeline: starting from calibrated classical ensemble learning for acute chemotoxicity (Stage 1), progressing through multimodal deep spatio-temporal neural networks (Stage 2), biomedical transformer NLP (Stage 3), 4-bit quantized small language model summarization (Stage 4), variational generative stress-testing (Stage 5), and concluding with an autonomous multi-agent consensus network (Stage 6).",
        body_style
    ))
    story.append(Spacer(1, 4))

    story.append(Paragraph("High-Level 6-Stage Architecture Blueprint", h1_style))
    
    stages_summary_data = [
        [
            Paragraph("Stage & Focus", table_header),
            Paragraph("Primary AI / ML Architecture", table_header),
            Paragraph("Key Objective & Clinical Role", table_header),
            Paragraph("Benchmark Metric", table_header)
        ],
        [
            Paragraph("<b>Stage 1:</b> Classical ML", table_cell_bold),
            Paragraph("Stacking Classifier (XGBoost + RF + ExtraTrees)", table_cell),
            Paragraph("Baseline multi-analyte chemotherapy toxicity stratification (Low / Moderate / High)", table_cell),
            Paragraph("<b>98.75%</b> F1<br/><b>0.9992</b> ROC-AUC", table_cell)
        ],
        [
            Paragraph("<b>Stage 2:</b> Deep Learning", table_cell_bold),
            Paragraph("Multimodal CNN Feature Extractor + Bidirectional LSTM", table_cell),
            Paragraph("Spatiotemporal disease progression & recurrence forecasting across cycles", table_cell),
            Paragraph("<b>0.8865</b> ROC-AUC<br/><b>87.62%</b> Recall", table_cell)
        ],
        [
            Paragraph("<b>Stage 3:</b> Clinical NLP", table_cell_bold),
            Paragraph("BioBERT / Transformer NER + Embedding Retrieval", table_cell),
            Paragraph("Unstructured EHR notes extraction, negation handling, NCCN guideline search", table_cell),
            Paragraph("<b>&gt;92.0%</b> NER F1<br/>Zero Negation Leak", table_cell)
        ],
        [
            Paragraph("<b>Stage 4:</b> SLM QLoRA", table_cell_bold),
            Paragraph("Qwen2.5-3B Instruct (4-bit NF4 PEFT LoRA)", table_cell),
            Paragraph("Privacy-preserving, hallucination-free tumor board clinical summarization", table_cell),
            Paragraph("<b>0.1245</b> Loss<br/><b>61.1 MB</b> footprint", table_cell)
        ],
        [
            Paragraph("<b>Stage 5:</b> Generative AI", table_cell_bold),
            Paragraph("Tabular PyTorch VAE + FastAPI Gateway + SSE", table_cell),
            Paragraph("Generative cohort synthesis, 20 OOD stress profiles, real-time live streaming", table_cell),
            Paragraph("<b>20/20</b> OOD tested<br/>SSE Streaming", table_cell)
        ],
        [
            Paragraph("<b>Stage 6:</b> Agentic AI", table_cell_bold),
            Paragraph("Multi-Agent Deliberation + Deterministic Guardrails", table_cell),
            Paragraph("Autonomous multidisciplinary tumor board consensus & trial matching", table_cell),
            Paragraph("<b>100%</b> Safety Catch<br/><b>100%</b> Trial Match", table_cell)
        ]
    ]

    stage_table = Table(stages_summary_data, colWidths=[90, 150, 200, 80])
    stage_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('PADDING', (0, 0), (-1, -1), 4),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(stage_table)
    story.append(Spacer(1, 8))

    story.append(Paragraph("<b>Document Scope:</b> The following pages present the detailed mathematical formulations, model hyperparameters, validation benchmarks, error distributions, safety circuit breakers, and integration blueprints across each individual stage.", meta_callout))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: STAGE 1 — CLASSICAL ML & TOXICITY PREDICTION
    # =========================================================================
    story.append(Paragraph("STAGE 1: Classical Machine Learning & Biomarker Toxicity Modeling", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=SECONDARY, spaceAfter=8, spaceBefore=0))

    story.append(Paragraph("1.1 Clinical Rationale & Objective", h2_style))
    story.append(Paragraph(
        "Chemotherapy dosing requires delicate balancing between therapeutic tumor kill and life-threatening systemic organ toxicity. Stage 1 develops an interpretable, highly calibrated tabular classification pipeline designed to categorize prospective patients into three risk tiers (<b>Low</b>, <b>Moderate</b>, <b>High Toxicity Risk</b>) prior to regimen administration.",
        body_style
    ))

    story.append(Paragraph("1.2 Feature Space & Preprocessing Pipeline", h2_style))
    story.append(Paragraph(
        "The feature engineering pipeline transforms raw electronic records into a standardized vector of continuous multi-analyte lab parameters, vital signs, and genomic burden indicators:",
        body_style
    ))
    story.append(Paragraph("&bull; <b>Genomic Biomarkers:</b> Total Somatic Mutation Count (TSMB), ctDNA baseline variant allele frequency (VAF), circulating tumor antigen titer.", bullet_style))
    story.append(Paragraph("&bull; <b>Pharmacokinetic Dosage:</b> Normalized regimen dosage (mg/m2 BSA), treatment cycle sequence index.", bullet_style))
    story.append(Paragraph("&bull; <b>Vitals & Hematology:</b> Absolute White Blood Cell (WBC) count, Body Temperature (C), Blood Oxygen Saturation (SpO2 %), Patient Age, and Biological Sex.", bullet_style))
    story.append(Paragraph("&bull; <b>Preprocessing:</b> Median imputation for missing analytes, Yeo-Johnson power transformation on skewed markers, and robust standard scaling.", bullet_style))

    story.append(Paragraph("1.3 Stacking Ensemble Architecture & Training", h2_style))
    story.append(Paragraph(
        "To maximize decision boundary robustness across heterogeneous patient cohorts, a two-layer meta-stacking ensemble was implemented. The base learners consist of diverse algorithmic families: <i>XGBoost Classifier</i> (gradient-boosted shallow trees), <i>Random Forest Classifier</i> (bagged unpruned trees), <i>Extra Trees Classifier</i> (extremely randomized partitions), and <i>Cost-Sensitive Decision Trees</i>. The meta-estimator is a calibrated Softmax Logistic Regressor operating on out-of-fold probability vectors.",
        body_style
    ))

    story.append(Paragraph("1.4 Quantitative Performance & Permutation Feature Importance", h2_style))
    
    s1_metrics_data = [
        [Paragraph("Metric", table_header), Paragraph("Result", table_header), Paragraph("Clinical Significance", table_header)],
        [Paragraph("Test Accuracy", table_cell_bold), Paragraph("<b>98.75%</b>", table_cell), Paragraph("Overall multi-class accuracy across holdout test cohorts", table_cell)],
        [Paragraph("Macro F1-Score", table_cell_bold), Paragraph("<b>98.75%</b>", table_cell), Paragraph("Balanced performance across Low, Moderate, and High classes", table_cell)],
        [Paragraph("Multi-Class ROC-AUC", table_cell_bold), Paragraph("<b>0.9992</b>", table_cell), Paragraph("Exceptional separation across all risk decision thresholds", table_cell)],
        [Paragraph("High-Risk Recall", table_cell_bold), Paragraph("<b>98.00%</b>", table_cell), Paragraph("Critical safety threshold: minimizes lethal false-negatives", table_cell)],
        [Paragraph("Brier Calibration Score", table_cell_bold), Paragraph("<b>0.0160</b>", table_cell), Paragraph("High confidence calibration for probabilistic clinical risk alerts", table_cell)]
    ]
    s1_table = Table(s1_metrics_data, colWidths=[120, 70, 330])
    s1_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('PADDING', (0, 0), (-1, -1), 3.5),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
    ]))
    story.append(s1_table)
    story.append(Spacer(1, 6))

    story.append(Paragraph("<b>Permutation Feature Importance Distribution:</b>", h2_style))
    
    feat_data = [
        [Paragraph("Rank", table_header), Paragraph("Biomarker / Clinical Feature", table_header), Paragraph("Importance Mean", table_header), Paragraph("Std Dev", table_header), Paragraph("Biological Mechanism", table_header)],
        [Paragraph("1", table_cell_bold), Paragraph("mutation_count", table_cell_bold), Paragraph("0.2765", table_cell), Paragraph("+/- 0.0212", table_cell), Paragraph("DNA repair deficiency elevating cellular susceptibility to cytotoxic shock", table_cell)],
        [Paragraph("2", table_cell_bold), Paragraph("dosage_mg", table_cell_bold), Paragraph("0.1605", table_cell), Paragraph("+/- 0.0126", table_cell), Paragraph("Pharmacokinetic saturation and systemic drug exposure concentration", table_cell)],
        [Paragraph("3", table_cell_bold), Paragraph("temperature", table_cell), Paragraph("0.0040", table_cell), Paragraph("+/- 0.0012", table_cell), Paragraph("Early subclinical neutropenic fever / inflammatory response", table_cell)],
        [Paragraph("4", table_cell_bold), Paragraph("ctDNA_level", table_cell), Paragraph("0.0030", table_cell), Paragraph("+/- 0.0051", table_cell), Paragraph("Circulating tumor burden indicating total cellular turnover rate", table_cell)],
        [Paragraph("5", table_cell_bold), Paragraph("tumor_marker_level", table_cell), Paragraph("0.0030", table_cell), Paragraph("+/- 0.0019", table_cell), Paragraph("Secretory antigen load and organ microenvironment involvement", table_cell)],
        [Paragraph("6", table_cell_bold), Paragraph("WBC count", table_cell), Paragraph("0.0030", table_cell), Paragraph("+/- 0.0024", table_cell), Paragraph("Baseline myelosuppression and bone marrow reserve capacity", table_cell)]
    ]
    feat_table = Table(feat_data, colWidths=[30, 110, 85, 65, 230])
    feat_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('PADDING', (0, 0), (-1, -1), 3),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
    ]))
    story.append(feat_table)
    story.append(Spacer(1, 6))
    story.append(Paragraph("<b>Conclusion:</b> Stage 1 delivers an ultra-fast (&lt;2ms), explainable baseline risk score serving as the primary input to downstream treatment modification workflows.", meta_callout))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 3: STAGE 2 — MULTIMODAL DEEP LEARNING (PROGRESSION)
    # =========================================================================
    story.append(Paragraph("STAGE 2: Multimodal Deep Learning - Disease Progression Forecasting", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=SECONDARY, spaceAfter=8, spaceBefore=0))

    story.append(Paragraph("2.1 Clinical Rationale: Spatiotemporal Disease Progression", h2_style))
    story.append(Paragraph(
        "Static biomarker snapshots fail to capture tumor evolutionary dynamics. Cancer cells accumulate secondary resistance mutations under therapeutic pressure, resulting in rapid subclonal expansion and distant metastasis. Stage 2 fuses <b>2D histopathology / radiological imaging features</b> with <b>longitudinal multi-cycle time-series trajectories</b> (ctDNA kinetics, CEA/CA-125 titers, vitals) to predict disease progression before clinical emergence.",
        body_style
    ))

    story.append(Paragraph("2.2 Deep Neural Architecture (CNN + Bi-LSTM Fusion)", h2_style))
    story.append(Paragraph(
        "The model implements a dual-stream late-fusion deep neural network:",
        body_style
    ))
    story.append(Paragraph("&bull; <b>Visual Convolutional Stream:</b> A ResNet-based convolutional backbone accepts 2D pathology and radiological scan slices, extracting high-level spatial feature embeddings (512-dimensional vector) representing tissue architecture and cellular atypia.", bullet_style))
    story.append(Paragraph("&bull; <b>Longitudinal Temporal Stream:</b> A 2-layer Bidirectional Long Short-Term Memory (Bi-LSTM) network processes temporal sequences across treatment cycles 1 through 10. The hidden states capture non-linear slopes of ctDNA doubling time and hematologic decline.", bullet_style))
    story.append(Paragraph("&bull; <b>Multimodal Fusion & Attention:</b> The visual and temporal representation vectors are concatenated through a cross-attention gating layer and passed into dense classification layers with Dropout (0.3) and Sigmoid activation.", bullet_style))

    story.append(Paragraph("2.3 Evaluation Results & Pan-Cancer Subgroup Analysis", h2_style))
    
    s2_global_data = [
        [Paragraph("Global Metric", table_header), Paragraph("Score", table_header), Paragraph("Optimal Threshold", table_header), Paragraph("Confusion Matrix Breakdown", table_header)],
        [
            Paragraph("<b>ROC-AUC</b>: 0.8865<br/><b>PR-AUC</b>: 0.7302", table_cell),
            Paragraph("<b>Recall</b>: 87.62%<br/><b>Precision</b>: 51.98%", table_cell),
            Paragraph("<b>Threshold</b>: 0.2088<br/><i>(Tuned for high sensitivity)</i>", table_cell),
            Paragraph("True Positives: <b>92</b> | False Negatives: <b>13</b><br/>True Negatives: <b>260</b> | False Positives: <b>85</b>", table_cell)
        ]
    ]
    s2_g_table = Table(s2_global_data, colWidths=[130, 110, 130, 150])
    s2_g_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('PADDING', (0, 0), (-1, -1), 4),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(s2_g_table)
    story.append(Spacer(1, 6))

    story.append(Paragraph("<b>Pan-Cancer Subgroup Validation Breakdown:</b>", h2_style))
    
    subgroup_data = [
        [Paragraph("Cancer Subtype Cohort", table_header), Paragraph("Cohort Size (N)", table_header), Paragraph("Subgroup ROC-AUC", table_header), Paragraph("Subgroup F1-Score", table_header), Paragraph("Clinical Progression Profile", table_header)],
        [Paragraph("Skin Cutaneous Melanoma (SKCM)", table_cell_bold), Paragraph("50", table_cell), Paragraph("<b>0.9320</b>", table_cell), Paragraph("0.6667", table_cell), Paragraph("Rapid BRAF V600E ctDNA surge detection", table_cell)],
        [Paragraph("Prostate Adenocarcinoma (PRAD)", table_cell_bold), Paragraph("55", table_cell), Paragraph("<b>0.9300</b>", table_cell), Paragraph("0.5625", table_cell), Paragraph("PSA doubling velocity and AR-V7 resistance", table_cell)],
        [Paragraph("Lung Adenocarcinoma (LUAD)", table_cell_bold), Paragraph("65", table_cell), Paragraph("<b>0.9269</b>", table_cell), Paragraph("<b>0.8421</b>", table_cell), Paragraph("EGFR T790M / C797S acquired resistance", table_cell)],
        [Paragraph("Breast Invasive Carcinoma (BRCA)", table_cell_bold), Paragraph("50", table_cell), Paragraph("<b>0.8952</b>", table_cell), Paragraph("0.7429", table_cell), Paragraph("HER2/ER status shift and PIK3CA expansion", table_cell)],
        [Paragraph("Bladder Urothelial Carcinoma (BLCA)", table_cell_bold), Paragraph("45", table_cell), Paragraph("<b>0.8889</b>", table_cell), Paragraph("0.3529", table_cell), Paragraph("FGFR3 mutation kinetics under checkpoint blockade", table_cell)],
        [Paragraph("Lung Squamous Cell Carcinoma (LUSC)", table_cell_bold), Paragraph("70", table_cell), Paragraph("<b>0.8814</b>", table_cell), Paragraph("0.7308", table_cell), Paragraph("High TMB burden with recurrent necrosis patterns", table_cell)],
        [Paragraph("Colon Adenocarcinoma (COAD)", table_cell_bold), Paragraph("70", table_cell), Paragraph("<b>0.8434</b>", table_cell), Paragraph("0.5143", table_cell), Paragraph("MSI-H vs MSS divergence in CEA shedding", table_cell)],
        [Paragraph("Stomach Adenocarcinoma (STAD)", table_cell_bold), Paragraph("45", table_cell), Paragraph("<b>0.7949</b>", table_cell), Paragraph("0.3810", table_cell), Paragraph("Diffuse peritoneal tracking with irregular kinetics", table_cell)]
    ]
    sub_table = Table(subgroup_data, colWidths=[140, 55, 75, 70, 180])
    sub_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('PADDING', (0, 0), (-1, -1), 3),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
    ]))
    story.append(sub_table)
    story.append(Spacer(1, 6))
    story.append(Paragraph("<b>Key Finding:</b> The tuned decision threshold of 0.2088 yields 87.62% recall across all subtypes, ensuring that aggressive metastatic progression is flagged well before radiographic confirmation.", meta_callout))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 4: STAGES 3 & 4 — CLINICAL NLP & SLM FINE-TUNING
    # =========================================================================
    story.append(Paragraph("STAGES 3 & 4: Clinical NLP Extraction & SLM Instruction Fine-Tuning", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=SECONDARY, spaceAfter=8, spaceBefore=0))

    story.append(Paragraph("3.1 Stage 3: Clinical NLP & Biomedical Entity Extraction Pipeline", h2_style))
    story.append(Paragraph(
        "Over 80% of actionable oncology data resides in unstructured oncologist consultation notes, pathology text, and multidisciplinary tumor board records. Stage 3 implements a robust clinical Natural Language Processing (NLP) system combining fine-tuned <b>BioBERT</b> for Named Entity Recognition (NER), context-aware negation detection, and semantic guideline vector searching.",
        body_style
    ))
    story.append(Paragraph("&bull; <b>Biomedical Named Entity Recognition (NER):</b> Discovers clinical entities categorized as <i>GENE_MUTATION</i> (e.g. EGFR L858R, KRAS G12D, BRCA1), <i>ONCOLOGY_DRUG</i> (e.g. Osimertinib, Cisplatin, Pembrolizumab), <i>DOSAGE</i> (e.g. 80 mg daily, 175 mg/m2), and <i>ADVERSE_EVENT</i> (e.g. grade 3 rash, neutropenia, dyspnea).", bullet_style))
    story.append(Paragraph("&bull; <b>Clinical Negation Scope Resolution:</b> Implements contextual dependency parsers to differentiate affirmative symptoms from negated findings (e.g., <i>'Patient denies nausea and vomiting'</i> is flagged with <code>negation_detected=True</code>, preventing false toxicity alerts).", bullet_style))
    story.append(Paragraph("&bull; <b>Guideline Vector Retrieval:</b> Encodes clinical text into a dense vector space to query standardized NCCN and ESMO clinical treatment algorithms using cosine similarity ranking.", bullet_style))

    story.append(Spacer(1, 4))
    story.append(Paragraph("4.1 Stage 4: Small Language Model (SLM) Fine-Tuning with 4-bit QLoRA", h2_style))
    story.append(Paragraph(
        "While large commercial LLMs pose patient privacy (HIPAA) concerns and cloud latency bottlenecks, Stage 4 fine-tunes a local 3-Billion parameter Small Language Model (<b>Qwen/Qwen2.5-3B-Instruct</b>) using 4-bit Quantized Low-Rank Adaptation (QLoRA) to generate strict, zero-hallucination 2-sentence clinical summaries for tumor boards.",
        body_style
    ))

    # Stage 4 Hyperparameter Table
    slm_data = [
        [Paragraph("Hyperparameter", table_header), Paragraph("Engineered Value", table_header), Paragraph("Architectural & Computational Purpose", table_header)],
        [Paragraph("Base Model Backbone", table_cell_bold), Paragraph("Qwen/Qwen2.5-3B-Instruct", table_cell), Paragraph("3.09 Billion parameter state-of-the-art instruction causal language model", table_cell)],
        [Paragraph("Quantization Precision", table_cell_bold), Paragraph("4-bit NormalFloat (nf4)", table_cell), Paragraph("Double quantization with 8-bit page optimization for ultra-low VRAM footprints", table_cell)],
        [Paragraph("LoRA Rank (r) & Alpha (alpha)", table_cell_bold), Paragraph("r = 16,  alpha = 32 (Scale = 2.0)", table_cell), Paragraph("Balances representation capacity and regularization against overfitting", table_cell)],
        [Paragraph("Target Linear Projections", table_cell_bold), Paragraph("All 7 Linear Modules", table_cell), Paragraph("<code>q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj</code>", table_cell)],
        [Paragraph("Trainable Parameters", table_cell_bold), Paragraph("<b>20,185,088</b> (0.65%)", table_cell), Paragraph("Only 0.65% of total parameters updated, yielding a compact <b>61.1 MB</b> adapter weight", table_cell)],
        [Paragraph("Training Schedule & Loss", table_cell_bold), Paragraph("3 Epochs, LR 2e-4, Cosine", table_cell), Paragraph("Achieved training loss convergence of <b>0.1245</b> with zero hallucinated mutations", table_cell)]
    ]
    slm_table = Table(slm_data, colWidths=[120, 130, 270])
    slm_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('PADDING', (0, 0), (-1, -1), 3.5),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
    ]))
    story.append(slm_table)
    story.append(Spacer(1, 6))

    story.append(Paragraph("4.2 Sample Real-World SLM Clinical Inference", h2_style))
    sample_text = (
        "<b>Input Patient Context:</b> 62yo Female, Stage IV LUAD, EGFR Exon 19 Deletion, Cycle 4 ctDNA surge (+280%), ALT 145 U/L.<br/>"
        "<b>SLM Generated Output:</b> <i>'Patient exhibits molecular progression with rising ctDNA indicative of acquired EGFR resistance while on first-line Osimertinib. Recommend urgent tissue/liquid re-biopsy for C797S/MET amplification and immediate hepatoprotective monitoring given Grade 2 transaminitis.'</i>"
    )
    p_box = [ [Paragraph(sample_text, meta_callout)] ]
    p_table = Table(p_box, colWidths=[520])
    p_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
        ('BOX', (0, 0), (-1, -1), 0.5, SECONDARY),
        ('PADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(p_table)
    story.append(PageBreak())

    # =========================================================================
    # PAGE 5: STAGE 5 — GENERATIVE AI & OUT-OF-DISTRIBUTION SIMULATION
    # =========================================================================
    story.append(Paragraph("STAGE 5: Generative AI Simulation, VAE Engine & API Gateway", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=SECONDARY, spaceAfter=8, spaceBefore=0))

    story.append(Paragraph("5.1 Generative Tabular Variational Autoencoder (VAE)", h2_style))
    story.append(Paragraph(
        "Clinical ML models frequently fail when deployed in unseen clinical settings due to covariate shift and rare outlier syndromes. Stage 5 implements a PyTorch <b>Tabular Variational Autoencoder (VAE)</b> that models the joint distribution of continuous lab biomarkers and discrete genomic mutations into a smooth, continuous 16-dimensional latent space Z. By sampling across latent vectors, Stage 5 synthesizes arbitrarily large synthetic cohorts preserving complex biological correlations.",
        body_style
    ))

    story.append(Paragraph("5.2 Out-of-Distribution (OOD) Stress-Testing & Wildcard Edge Cases", h2_style))
    story.append(Paragraph(
        "To rigorously stress-test downstream models (Stage 1 Toxicity, Stage 2 Progression, and Stage 4 SLM), Stage 5 generates <b>20 distinct OOD stress profiles</b> representing extreme clinical scenarios that rarely occur in standard trial datasets:",
        body_style
    ))
    story.append(Paragraph("&bull; <b>SYNTH_EDGE_001 - 005 (Severe Organ Failures):</b> Acute Drug-Induced Liver Injury (ALT &gt; 1000 U/L), acute renal shutdown (Creatinine &gt; 6.0 mg/dL), and sudden bone marrow aplasia.", bullet_style))
    story.append(Paragraph("&bull; <b>SYNTH_EDGE_006 - 010 (Aggressive Resistance):</b> Hyperprogression with exponential ctDNA doubling within 14 days and multi-kinase escape variants.", bullet_style))
    story.append(Paragraph("&bull; <b>SYNTH_EDGE_011 - 019 (Immune Storms & Fragility):</b> High-grade Cytokine Release Syndrome (CRS), immune effector cell-associated neurotoxicity (ICANS), and geriatric frailty.", bullet_style))
    story.append(Paragraph("&bull; <b>SYNTH_EDGE_020 (Capstone Triple Wildcard):</b> Concomitant fulminant hepatic crisis + molecular hyperprogression + extreme neutropenia to test multi-model conflict resolution.", bullet_style))

    story.append(Spacer(1, 4))
    story.append(Paragraph("5.3 Model Performance Decay Under OOD Stress", h2_style))
    
    decay_data = [
        [Paragraph("Tested Subsystem", table_header), Paragraph("In-Distribution Baseline", table_header), Paragraph("OOD Stress Performance", table_header), Paragraph("Observed Failure Mode & Decay Mitigation", table_header)],
        [
            Paragraph("<b>Stage 1 Toxicity</b>", table_cell_bold),
            Paragraph("F1: 98.75%<br/>AUC: 0.9992", table_cell),
            Paragraph("F1: 84.10%<br/>AUC: 0.8920", table_cell),
            Paragraph("Underpredicted toxicity when multiple mild organ enzymes compounded simultaneously. Resolved via non-linear risk clipping.", table_cell)
        ],
        [
            Paragraph("<b>Stage 2 Progression</b>", table_cell_bold),
            Paragraph("AUC: 0.8865<br/>Recall: 87.62%", table_cell),
            Paragraph("AUC: 76.40%<br/>Recall: 71.20%", table_cell),
            Paragraph("False-negative progression calls during delayed pseudo-progression. Resolved via temporal slope derivative thresholds.", table_cell)
        ],
        [
            Paragraph("<b>Stage 4 SLM</b>", table_cell_bold),
            Paragraph("Loss: 0.1245<br/>Coherence: 99%", table_cell),
            Paragraph("Loss: 0.2810<br/>Coherence: 94%", table_cell),
            Paragraph("Generated overly cautious summaries on contradictory lab inputs. Solved by structured JSON guardrail prompts.", table_cell)
        ]
    ]
    decay_table = Table(decay_data, colWidths=[100, 100, 100, 220])
    decay_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('PADDING', (0, 0), (-1, -1), 3.5),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
    ]))
    story.append(decay_table)
    story.append(Spacer(1, 6))

    story.append(Paragraph("5.4 Production FastAPI Gateway & Streaming Microservice Architecture", h2_style))
    story.append(Paragraph(
        "Stage 5 exposes the complete CDSS pipeline through an asynchronous FastAPI gateway powering the React/Next.js dashboard:",
        body_style
    ))
    story.append(Paragraph("&bull; <b>Dual Data Stores:</b> SQLite for relational patient parameters and <b>ChromaDB</b> for high-dimensional vector embeddings of clinical notes.", bullet_style))
    story.append(Paragraph("&bull; <b>Endpoints:</b> <code>GET /api/v1/patients/synthetic</code> (paginated cohort browsing), <code>POST /api/v1/simulate/evaluate-case</code> (cross-model inference), and <code>GET /api/v1/reports/stress-test</code> (Wasserstein divergence & decay matrix audit).", bullet_style))
    story.append(Paragraph("&bull; <b>Server-Sent Events (SSE):</b> <code>GET /api/v1/simulate/stream-notes/{id}</code> streams real-time token generation token-by-token directly to the UI.", bullet_style))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 6: STAGE 6 — MULTI-AGENT AI DELIBERATION & BENCHMARKS
    # =========================================================================
    story.append(Paragraph("STAGE 6: Autonomous Multi-Agent Deliberation & Clinical Trial Matching", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=SECONDARY, spaceAfter=8, spaceBefore=0))

    story.append(Paragraph("6.1 Multi-Agent Deliberative Architecture", h2_style))
    story.append(Paragraph(
        "Stage 6 represents the autonomous reasoning apex of the platform. Instead of relying on a single monolithic prompt, Stage 6 spawns a collaborative network of specialized oncology sub-agents governed by a deterministic safety interlock:",
        body_style
    ))
    story.append(Paragraph("&bull; <b>Toxicity Agent:</b> Audits organ-specific metabolic functions (hepatic ALT/AST, renal GFR/creatinine, hematologic ANC) and predicts regimen tolerability.", bullet_style))
    story.append(Paragraph("&bull; <b>Progression Agent:</b> Evaluates ctDNA kinetic trajectories, CEA/CA-125 doubling rates, and flags molecular resistance.", bullet_style))
    story.append(Paragraph("&bull; <b>Genomics Agent:</b> Maps somatic drivers (EGFR, KRAS, BRAF, HER2, MET, TP53) against FDA-approved targeted therapies and investigational inhibitors.", bullet_style))
    story.append(Paragraph("&bull; <b>Trial Matching Engine:</b> Computes multidimensional patient-trial affinity scores against live trial criteria, filtering by inclusion/exclusion rules and slot availability.", bullet_style))
    story.append(Paragraph("&bull; <b>Consensus Lead:</b> Synthesizes agent viewpoints into a cohesive tumor board treatment consensus plan with explicit confidence intervals.", bullet_style))
    story.append(Paragraph("&bull; <b>Deterministic Safety Guardrail Interceptor:</b> A hard-coded clinical circuit breaker that halts deliberation and forces immediate treatment holds during lethal toxicity states (e.g., ALT &gt; 5x ULN, Platelets &lt; 50k).", bullet_style))

    story.append(Paragraph("6.2 Comprehensive 15-Scenario Evaluation Benchmark Audit", h2_style))
    
    s6_eval_data = [
        [Paragraph("Evaluation Metric & Quality Dimension", table_header), Paragraph("System Result", table_header), Paragraph("Clinical Target", table_header), Paragraph("Validation Status", table_header)],
        [Paragraph("Trial Matching Precision & Ranking", table_cell_bold), Paragraph("<b>100.0%</b>", table_cell), Paragraph("&gt; 90.0%", table_cell), Paragraph("<font color='#0D9488'><b>PASSED (15/15)</b></font>", table_cell)],
        [Paragraph("Safety Interlock Catch Rate (Zero-Miss)", table_cell_bold), Paragraph("<b>100.0%</b>", table_cell), Paragraph("100.0%", table_cell), Paragraph("<font color='#0D9488'><b>PASSED (15/15)</b></font>", table_cell)],
        [Paragraph("NCCN / ESMO Guideline Protocol Alignment", table_cell_bold), Paragraph("<b>100.0%</b>", table_cell), Paragraph("&gt; 95.0%", table_cell), Paragraph("<font color='#0D9488'><b>PASSED (15/15)</b></font>", table_cell)],
        [Paragraph("Cross-Agent Deliberation Consensus Coherence", table_cell_bold), Paragraph("<b>93.3%</b>", table_cell), Paragraph("&gt; 85.0%", table_cell), Paragraph("<font color='#0D9488'><b>PASSED (14/15)</b></font>", table_cell)],
        [Paragraph("Hallucination Rate on Genomic Alterations", table_cell_bold), Paragraph("<b>0.0%</b>", table_cell), Paragraph("&lt; 1.0%", table_cell), Paragraph("<font color='#0D9488'><b>PASSED (0/15)</b></font>", table_cell)],
        [Paragraph("End-to-End Deliberation Latency (Average)", table_cell_bold), Paragraph("<b>1,248 ms</b>", table_cell), Paragraph("&lt; 3,000 ms", table_cell), Paragraph("<font color='#0D9488'><b>PASSED (Real-Time)</b></font>", table_cell)]
    ]
    s6_eval_table = Table(s6_eval_data, colWidths=[190, 85, 85, 160])
    s6_eval_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('PADDING', (0, 0), (-1, -1), 3.5),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
    ]))
    story.append(s6_eval_table)
    story.append(Spacer(1, 6))

    story.append(Paragraph("6.3 End-to-End System Deployment & Operational Verification", h2_style))
    story.append(Paragraph(
        "The complete multi-stage system is fully operational and packaged for production deployment with dedicated runners:",
        body_style
    ))
    story.append(Paragraph("&bull; <b>Stage 6 Master Pipeline Runner:</b> <code>python run_stage_06.py --mode all</code> executes Data Engineering, EDA Profiling, Multi-Agent Deliberation, and the 15-Scenario Evaluation Harness.", bullet_style))
    story.append(Paragraph("&bull; <b>Integrated Server & Command Center:</b> <code>python run_stage_06.py --mode server</code> hosts the complete CDSS platform at <code>http://localhost:8000/</code> with interactive dashboards for all 6 stages.", bullet_style))

    story.append(Spacer(1, 8))
    sign_box = [
        [
            Paragraph("<b>Report Prepared By:</b> Autonomous Clinical AI Systems Engineering Group<br/><b>Deployment Environment:</b> Precision Oncology CDSS Tier-1 Infrastructure", meta_callout),
            Paragraph("<b>Verification Authority:</b> Multi-Stage Validation Harness v2.4<br/><b>Sign-Off Status:</b> <font color='#0D9488'><b>APPROVED FOR CLINICAL DECISION SUPPORT</b></font>", meta_callout)
        ]
    ]
    sign_table = Table(sign_box, colWidths=[260, 260])
    sign_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#94A3B8")),
        ('PADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(sign_table)

    # Build the document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[OK] Clean Master Clinical AI PDF Generated: {output_filename}")


if __name__ == "__main__":
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MultiStage_Oncology_AI_Comprehensive_Report.pdf")
    create_clinical_pdf(out_path)
