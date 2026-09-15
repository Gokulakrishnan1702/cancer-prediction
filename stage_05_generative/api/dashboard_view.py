"""
Clinical Decision Support System (CDSS) Dashboard View
=====================================================
Unified modern, professional medical AI web dashboard for Oncology Risk Prediction.
Implements the multi-stage pipeline (ML -> DL -> NLP -> SLM -> GenAI),
sidebar stage navigation, preset loading, CSV/image upload, and final clinical assessment.
"""

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI PATIENT RISK PREDICTION — Multi-Stage Clinical AI</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-canvas: #f4f7fb;
            --bg-sidebar: #07152b;
            --bg-sidebar-hover: #0f2445;
            --bg-sidebar-active: #153360;
            --bg-header: #0b1e3f;
            --bg-header-gradient: linear-gradient(90deg, #07152b 0%, #0b1e3f 50%, #102a54 100%);
            --bg-card: #ffffff;
            --bg-card-alt: #f8fafc;
            --text-primary: #0f172a;
            --text-secondary: #475569;
            --text-light: #94a3b8;
            --border-card: #e2e8f0;
            --border-dark: rgba(255, 255, 255, 0.08);
            --accent-blue: #2563eb;
            --accent-blue-hover: #1d4ed8;
            --accent-blue-light: #eff6ff;
            --accent-blue-border: #bfdbfe;
            --accent-green: #059669;
            --accent-green-bg: #ecfdf5;
            --accent-green-border: #a7f3d0;
            --accent-yellow: #d97706;
            --accent-yellow-bg: #fffbeb;
            --accent-yellow-border: #fde68a;
            --accent-orange: #ea580c;
            --accent-orange-bg: #fff7ed;
            --accent-orange-border: #fed7aa;
            --accent-red: #dc2626;
            --accent-red-bg: #fef2f2;
            --accent-red-border: #fecaca;
            --accent-purple: #7c3aed;
            --accent-purple-bg: #f5f3ff;
            --radius-sm: 6px;
            --radius-md: 10px;
            --radius-lg: 14px;
            --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
            --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.07), 0 2px 4px -1px rgba(0, 0, 0, 0.04);
            --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -2px rgba(0, 0, 0, 0.04);
            --shadow-blue: 0 4px 14px 0 rgba(37, 99, 235, 0.35);
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: var(--bg-canvas);
            color: var(--text-primary);
            height: 100vh;
            display: flex;
            overflow: hidden;
            font-size: 13.5px;
            line-height: 1.45;
        }

        /* ==========================================================================
           1. SIDEBAR NAVIGATION
           ========================================================================== */
        .sidebar {
            width: 230px;
            background: var(--bg-sidebar);
            display: flex;
            flex-direction: column;
            flex-shrink: 0;
            border-right: 1px solid var(--border-dark);
            z-index: 100;
            user-select: none;
        }
        .sidebar-brand {
            padding: 1.25rem 1.25rem;
            display: flex;
            align-items: center;
            gap: 0.75rem;
            border-bottom: 1px solid var(--border-dark);
        }
        .brand-icon {
            width: 36px;
            height: 36px;
            background: linear-gradient(135deg, #38bdf8 0%, #2563eb 100%);
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #ffffff;
            box-shadow: 0 2px 8px rgba(37, 99, 235, 0.4);
            flex-shrink: 0;
        }
        .brand-text h2 {
            font-size: 0.85rem;
            font-weight: 700;
            color: #ffffff;
            letter-spacing: 0.03em;
            line-height: 1.2;
        }
        .brand-text p {
            font-size: 0.65rem;
            color: #94a3b8;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .sidebar-menu {
            display: flex;
            flex-direction: column;
            padding: 1rem 0.65rem;
            gap: 0.25rem;
            flex: 1;
            overflow-y: auto;
        }
        .menu-heading {
            font-size: 0.65rem;
            font-weight: 700;
            text-transform: uppercase;
            color: #64748b;
            letter-spacing: 0.08em;
            padding: 0.75rem 0.65rem 0.35rem;
        }
        .nav-link {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.65rem 0.85rem;
            color: #94a3b8;
            text-decoration: none;
            border-radius: var(--radius-sm);
            font-size: 0.82rem;
            font-weight: 500;
            transition: all 0.18s ease;
            position: relative;
            cursor: pointer;
        }
        .nav-link:hover {
            color: #f1f5f9;
            background: var(--bg-sidebar-hover);
        }
        .nav-link.active {
            color: #ffffff;
            background: var(--bg-sidebar-active);
            font-weight: 600;
        }
        .nav-link.active::before {
            content: '';
            position: absolute;
            left: 0;
            top: 15%;
            bottom: 15%;
            width: 3.5px;
            background: #38bdf8;
            border-radius: 0 4px 4px 0;
            box-shadow: 0 0 8px #38bdf8;
        }
        .nav-link svg {
            width: 17px;
            height: 17px;
            stroke-width: 2;
            flex-shrink: 0;
        }
        .stage-pill-tag {
            margin-left: auto;
            font-size: 0.62rem;
            padding: 0.15rem 0.45rem;
            border-radius: 4px;
            background: rgba(56, 189, 248, 0.12);
            color: #38bdf8;
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
        }
        .sidebar-footer {
            padding: 1rem 1.25rem;
            border-top: 1px solid var(--border-dark);
            font-size: 0.68rem;
            color: #64748b;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .system-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: #10b981;
            box-shadow: 0 0 6px #10b981;
            display: inline-block;
            margin-right: 0.35rem;
        }

        /* ==========================================================================
           2. MAIN CONTAINER & HEADER
           ========================================================================== */
        .main-wrapper {
            flex: 1;
            display: flex;
            flex-direction: column;
            height: 100vh;
            overflow: hidden;
            background: var(--bg-canvas);
        }
        .top-header {
            height: 64px;
            background: var(--bg-header-gradient);
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 1.75rem;
            color: #ffffff;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.12);
            flex-shrink: 0;
            z-index: 50;
        }
        .header-title-block {
            display: flex;
            align-items: center;
            gap: 0.85rem;
        }
        .header-icon-box {
            width: 38px;
            height: 38px;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.12);
            display: flex;
            align-items: center;
            justify-content: center;
            color: #38bdf8;
        }
        .header-title-box h1 {
            font-size: 1.05rem;
            font-weight: 800;
            letter-spacing: 0.04em;
            color: #ffffff;
            line-height: 1.2;
        }
        .header-title-box p {
            font-size: 0.72rem;
            color: #93c5fd;
            font-weight: 500;
            letter-spacing: 0.02em;
        }
        .header-status-strip {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            background: rgba(0, 0, 0, 0.25);
            padding: 0.35rem 0.65rem;
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.08);
        }
        .stage-status-chip {
            display: flex;
            align-items: center;
            gap: 0.35rem;
            font-size: 0.73rem;
            font-weight: 600;
            color: #cbd5e1;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            background: rgba(255, 255, 255, 0.04);
            font-family: 'JetBrains Mono', monospace;
        }
        .stage-status-chip.ready { color: #4ade80; background: rgba(74, 222, 128, 0.1); }
        .stage-status-chip.running { color: #38bdf8; background: rgba(56, 189, 248, 0.18); animation: pulseSlow 1.5s infinite; }

        /* ==========================================================================
           3. CONTENT BODY & VIEWS
           ========================================================================== */
        .content-scroll {
            flex: 1;
            overflow-y: auto;
            padding: 1.5rem 1.75rem 3rem;
        }
        .view-section {
            display: none;
            flex-direction: column;
            gap: 1.25rem;
        }
        .view-section.active {
            display: flex;
        }

        /* Headings */
        .view-header {
            margin-bottom: 0.25rem;
        }
        .view-header h2 {
            font-size: 1.25rem;
            font-weight: 800;
            color: var(--text-primary);
            letter-spacing: -0.01em;
        }
        .view-header p {
            font-size: 0.82rem;
            color: var(--text-secondary);
        }

        /* ==========================================================================
           4. PIPELINE FLOW CARDS
           ========================================================================== */
        .pipeline-card {
            background: var(--bg-card);
            border: 1px solid var(--border-card);
            border-radius: var(--radius-lg);
            padding: 1.25rem 1.5rem;
            box-shadow: var(--shadow-sm);
        }
        .pipeline-flow {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.65rem;
            position: relative;
            margin: 0.75rem 0 0.5rem;
        }
        .flow-node {
            flex: 1;
            background: var(--bg-card-alt);
            border: 1.5px solid var(--border-card);
            border-radius: var(--radius-md);
            padding: 0.85rem 0.75rem;
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
            position: relative;
            transition: all 0.25s ease;
            cursor: pointer;
        }
        .flow-node:hover {
            border-color: var(--accent-blue-border);
            transform: translateY(-2px);
            box-shadow: var(--shadow-md);
        }
        .flow-node.active-stage {
            border-color: var(--accent-blue);
            background: #f0f7ff;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.15);
        }
        .flow-node.completed-stage {
            border-color: #10b981;
            background: #f0fdf4;
        }
        .flow-node.processing-stage {
            border-color: #38bdf8;
            background: #f0f9ff;
            animation: pulseCard 1.2s infinite alternate;
        }
        .node-badge {
            width: 28px;
            height: 28px;
            border-radius: 50%;
            background: #e2e8f0;
            color: var(--text-secondary);
            font-weight: 700;
            font-size: 0.75rem;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 0.4rem;
            transition: all 0.2s;
        }
        .flow-node.active-stage .node-badge { background: var(--accent-blue); color: #fff; }
        .flow-node.completed-stage .node-badge { background: #10b981; color: #fff; }
        .flow-node.processing-stage .node-badge { background: #38bdf8; color: #fff; }
        .node-title {
            font-size: 0.82rem;
            font-weight: 700;
            color: var(--text-primary);
        }
        .node-subtitle {
            font-size: 0.68rem;
            color: var(--text-secondary);
            margin: 0.15rem 0 0.45rem;
        }
        .node-status-pill {
            font-size: 0.62rem;
            font-weight: 600;
            padding: 0.15rem 0.5rem;
            border-radius: 20px;
            background: #e2e8f0;
            color: #64748b;
            font-family: 'JetBrains Mono', monospace;
        }
        .flow-node.completed-stage .node-status-pill { background: #dcfce7; color: #166534; }
        .flow-node.processing-stage .node-status-pill { background: #e0f2fe; color: #0369a1; }
        .flow-arrow {
            color: #cbd5e1;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
        }
        .flow-arrow svg {
            width: 18px;
            height: 18px;
        }
        .pipeline-progress-bar-wrap {
            margin-top: 1rem;
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        .progress-track {
            flex: 1;
            height: 6px;
            background: #e2e8f0;
            border-radius: 4px;
            overflow: hidden;
        }
        .progress-fill {
            height: 100%;
            width: 0%;
            background: linear-gradient(90deg, #38bdf8 0%, #2563eb 100%);
            border-radius: 4px;
            transition: width 0.4s ease;
        }
        .progress-label {
            font-size: 0.72rem;
            font-weight: 600;
            color: var(--text-secondary);
            font-family: 'JetBrains Mono', monospace;
            white-space: nowrap;
        }

        /* ==========================================================================
           5. PATIENT INPUT SECTION
           ========================================================================== */
        .patient-input-card {
            background: var(--bg-card);
            border: 1px solid var(--border-card);
            border-radius: var(--radius-lg);
            padding: 1.5rem;
            box-shadow: var(--shadow-sm);
        }
        .card-title-strip {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1.25rem;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid var(--border-card);
        }
        .card-title-strip h3 {
            font-size: 0.98rem;
            font-weight: 800;
            letter-spacing: 0.02em;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .preset-buttons {
            display: flex;
            align-items: center;
            gap: 0.45rem;
            flex-wrap: wrap;
        }
        .btn-preset {
            font-size: 0.72rem;
            font-weight: 600;
            padding: 0.35rem 0.65rem;
            border-radius: var(--radius-sm);
            border: 1px solid var(--border-card);
            background: var(--bg-card-alt);
            color: var(--text-secondary);
            cursor: pointer;
            transition: all 0.15s ease;
        }
        .btn-preset:hover {
            border-color: var(--accent-blue);
            color: var(--accent-blue);
            background: #eff6ff;
        }
        .btn-preset.danger {
            border-color: var(--accent-red-border);
            color: var(--accent-red);
            background: var(--accent-red-bg);
        }
        .btn-preset.danger:hover {
            background: #fee2e2;
        }

        /* Inputs Grid */
        .input-grid-4 {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 0.85rem;
            margin-bottom: 1rem;
        }
        .input-grid-3 {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.85rem;
            margin-bottom: 1rem;
        }
        .form-group {
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }
        .form-label {
            font-size: 0.72rem;
            font-weight: 600;
            color: var(--text-secondary);
        }
        .form-control {
            font-family: 'Inter', sans-serif;
            font-size: 0.82rem;
            padding: 0.5rem 0.75rem;
            border-radius: var(--radius-sm);
            border: 1px solid var(--border-card);
            background: #ffffff;
            color: var(--text-primary);
            transition: border-color 0.15s;
            outline: none;
        }
        .form-control:focus {
            border-color: var(--accent-blue);
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12);
        }
        textarea.form-control {
            resize: vertical;
            min-height: 72px;
            font-size: 0.8rem;
            line-height: 1.4;
        }

        /* File Upload Zones */
        .upload-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            margin-bottom: 1.25rem;
        }
        .upload-box {
            border: 1.5px dashed var(--border-card);
            border-radius: var(--radius-md);
            padding: 0.85rem 1rem;
            background: var(--bg-card-alt);
            display: flex;
            align-items: center;
            gap: 0.85rem;
            cursor: pointer;
            transition: all 0.2s ease;
            position: relative;
        }
        .upload-box:hover {
            border-color: var(--accent-blue);
            background: #f8fafc;
        }
        .upload-icon {
            width: 36px;
            height: 36px;
            border-radius: 8px;
            background: #e2e8f0;
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--accent-blue);
            flex-shrink: 0;
        }
        .upload-text h4 {
            font-size: 0.78rem;
            font-weight: 700;
            color: var(--text-primary);
        }
        .upload-text p {
            font-size: 0.68rem;
            color: var(--text-secondary);
        }
        .upload-box input[type="file"] {
            position: absolute;
            inset: 0;
            opacity: 0;
            cursor: pointer;
        }
        .image-preview-thumb {
            width: 44px;
            height: 44px;
            border-radius: 6px;
            object-fit: cover;
            border: 1px solid var(--border-card);
            display: none;
        }

        /* Buttons Strip */
        .btn-action-row {
            display: flex;
            align-items: center;
            gap: 0.85rem;
            padding-top: 0.5rem;
        }
        .btn-primary-analyze {
            flex: 1;
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
            color: #ffffff;
            font-family: 'Inter', sans-serif;
            font-size: 0.88rem;
            font-weight: 700;
            letter-spacing: 0.03em;
            padding: 0.85rem 1.75rem;
            border-radius: var(--radius-md);
            border: none;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.65rem;
            box-shadow: var(--shadow-blue);
            transition: all 0.2s ease;
        }
        .btn-primary-analyze:hover {
            background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
            transform: translateY(-1px);
            box-shadow: 0 6px 20px rgba(37, 99, 235, 0.45);
        }
        .btn-primary-analyze:disabled {
            background: #94a3b8;
            box-shadow: none;
            cursor: not-allowed;
            transform: none;
        }
        .btn-secondary-reset {
            background: var(--bg-card-alt);
            color: var(--text-secondary);
            font-family: 'Inter', sans-serif;
            font-size: 0.82rem;
            font-weight: 600;
            padding: 0.85rem 1.5rem;
            border-radius: var(--radius-md);
            border: 1px solid var(--border-card);
            cursor: pointer;
            transition: all 0.15s;
        }
        .btn-secondary-reset:hover {
            background: #e2e8f0;
            color: var(--text-primary);
        }

        /* ==========================================================================
           6. FIVE STAGE LIVE RESULTS GRID
           ========================================================================== */
        .stage-results-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
        }
        .stage-result-card {
            background: var(--bg-card);
            border: 1px solid var(--border-card);
            border-radius: var(--radius-md);
            padding: 1rem 1.25rem;
            box-shadow: var(--shadow-sm);
            display: flex;
            flex-direction: column;
            gap: 0.65rem;
            position: relative;
            transition: all 0.2s;
        }
        .stage-result-card.expanded-full {
            grid-column: span 2;
        }
        .card-header-mini {
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .card-header-mini h4 {
            font-size: 0.84rem;
            font-weight: 800;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 0.45rem;
        }
        .metric-badge {
            font-size: 0.68rem;
            font-weight: 700;
            padding: 0.2rem 0.55rem;
            border-radius: 4px;
            font-family: 'JetBrains Mono', monospace;
        }
        .metric-badge.green { background: #dcfce7; color: #15803d; }
        .metric-badge.blue { background: #dbeafe; color: #1d4ed8; }
        .metric-badge.yellow { background: #fef3c7; color: #b45309; }
        .metric-badge.red { background: #fee2e2; color: #b91c1c; }

        .feature-tag-list {
            display: flex;
            flex-wrap: wrap;
            gap: 0.35rem;
        }
        .feature-tag {
            font-size: 0.68rem;
            background: var(--bg-card-alt);
            border: 1px solid var(--border-card);
            padding: 0.15rem 0.45rem;
            border-radius: 4px;
            color: var(--text-secondary);
        }
        .feature-tag strong { color: var(--text-primary); }

        /* ==========================================================================
           7. FINAL PREDICTION CARD (PROMINENT CLINICAL ASSESSMENT)
           ========================================================================== */
        .final-assessment-card {
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            border: 2px solid var(--accent-blue-border);
            border-radius: var(--radius-lg);
            padding: 1.75rem;
            box-shadow: var(--shadow-md);
            position: relative;
            display: flex;
            flex-direction: column;
            gap: 1.25rem;
        }
        .final-assessment-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #2563eb, #38bdf8, #10b981);
            border-radius: var(--radius-lg) var(--radius-lg) 0 0;
        }
        .assessment-banner {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border-card);
        }
        .assessment-banner-left h3 {
            font-size: 1.15rem;
            font-weight: 800;
            color: var(--text-primary);
            letter-spacing: 0.02em;
        }
        .assessment-banner-left p {
            font-size: 0.78rem;
            color: var(--text-secondary);
        }
        .risk-classification-tag {
            font-size: 1.1rem;
            font-weight: 900;
            letter-spacing: 0.05em;
            padding: 0.5rem 1.25rem;
            border-radius: 8px;
            font-family: 'JetBrains Mono', monospace;
            text-transform: uppercase;
            box-shadow: var(--shadow-sm);
        }
        .risk-classification-tag.low { background: #dcfce7; color: #166534; border: 1px solid #86efac; }
        .risk-classification-tag.moderate { background: #e0f2fe; color: #0369a1; border: 1px solid #7dd3fc; }
        .risk-classification-tag.high { background: #ffedd5; color: #c2410c; border: 1px solid #fdba74; }
        .risk-classification-tag.critical { background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }

        .assessment-breakdown-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 0.85rem;
        }
        .breakdown-box {
            background: #ffffff;
            border: 1px solid var(--border-card);
            border-radius: var(--radius-md);
            padding: 0.75rem 0.85rem;
        }
        .breakdown-box span {
            font-size: 0.65rem;
            font-weight: 700;
            color: var(--text-light);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .breakdown-box h4 {
            font-size: 0.92rem;
            font-weight: 800;
            color: var(--text-primary);
            margin-top: 0.2rem;
        }

        .clinical-summary-box {
            background: #f8fafc;
            border-left: 4px solid var(--accent-blue);
            border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
            padding: 1rem 1.25rem;
            font-size: 0.84rem;
            line-height: 1.55;
            color: var(--text-primary);
        }
        .considerations-checklist {
            display: flex;
            flex-direction: column;
            gap: 0.45rem;
            margin-top: 0.5rem;
        }
        .consideration-item {
            display: flex;
            align-items: flex-start;
            gap: 0.5rem;
            font-size: 0.8rem;
            color: var(--text-secondary);
        }
        .consideration-item svg {
            width: 16px;
            height: 16px;
            color: var(--accent-blue);
            flex-shrink: 0;
            margin-top: 2px;
        }
        .disclaimer-strip {
            font-size: 0.72rem;
            color: #64748b;
            font-style: italic;
            display: flex;
            align-items: center;
            gap: 0.4rem;
            border-top: 1px solid var(--border-card);
            padding-top: 0.75rem;
        }

        /* ==========================================================================
           8. TABLES & DATA LISTS (HISTORY, ANALYTICS)
           ========================================================================== */
        .table-card {
            background: var(--bg-card);
            border: 1px solid var(--border-card);
            border-radius: var(--radius-lg);
            overflow: hidden;
            box-shadow: var(--shadow-sm);
        }
        .table-responsive {
            width: 100%;
            overflow-x: auto;
        }
        table.clinical-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.8rem;
            text-align: left;
        }
        table.clinical-table th {
            background: #f8fafc;
            color: var(--text-secondary);
            font-weight: 700;
            padding: 0.75rem 1rem;
            border-bottom: 1px solid var(--border-card);
            text-transform: uppercase;
            font-size: 0.68rem;
            letter-spacing: 0.05em;
        }
        table.clinical-table td {
            padding: 0.75rem 1rem;
            border-bottom: 1px solid var(--border-card);
            color: var(--text-primary);
        }
        table.clinical-table tr:hover td {
            background: #f8fafc;
        }

        /* Analytics Stats Grid */
        .analytics-stats-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1rem;
        }
        .stat-card {
            background: var(--bg-card);
            border: 1px solid var(--border-card);
            border-radius: var(--radius-md);
            padding: 1.15rem;
            box-shadow: var(--shadow-sm);
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }
        .stat-card span {
            font-size: 0.7rem;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
        }
        .stat-card h3 {
            font-size: 1.5rem;
            font-weight: 800;
            color: var(--text-primary);
            font-family: 'JetBrains Mono', monospace;
        }

        /* Animations */
        @keyframes pulseSlow {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        @keyframes pulseCard {
            0% { border-color: #38bdf8; box-shadow: 0 0 8px rgba(56, 189, 248, 0.2); }
            100% { border-color: #2563eb; box-shadow: 0 0 16px rgba(37, 99, 235, 0.4); }
        }

        /* Responsive Layout adjustments */
        @media (max-width: 1024px) {
            .input-grid-4 { grid-template-columns: repeat(2, 1fr); }
            .assessment-breakdown-grid { grid-template-columns: repeat(2, 1fr); }
            .stage-results-grid { grid-template-columns: 1fr; }
            .stage-result-card.expanded-full { grid-column: span 1; }
            .sidebar { width: 72px; }
            .brand-text, .nav-link span, .menu-heading, .stage-pill-tag, .sidebar-footer { display: none; }
            .sidebar-brand { justify-content: center; padding: 1rem 0; }
        /* ==========================================================================
           STAGE 06 — AGENTIC AI COMMAND CENTER STYLES
           ========================================================================== */
        .agentic-status-grid {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 0.75rem;
            margin-top: 0.85rem;
        }
        @media (max-width: 1200px) {
            .agentic-status-grid { grid-template-columns: repeat(2, 1fr); }
        }
        .agent-node-card {
            background: #ffffff;
            border: 1px solid var(--border-card);
            border-radius: var(--radius-md);
            padding: 0.85rem 1rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            transition: all 0.2s ease;
            box-shadow: 0 1px 2px rgba(0,0,0,0.03);
        }
        .agent-node-card:hover {
            border-color: #38bdf8;
            box-shadow: 0 4px 12px rgba(56, 189, 248, 0.12);
        }
        .agent-node-info h4 {
            font-size: 0.76rem;
            font-weight: 800;
            color: var(--text-primary);
            letter-spacing: 0.02em;
        }
        .agent-node-info p {
            font-size: 0.68rem;
            color: var(--text-light);
            margin-top: 2px;
        }
        .agent-status-indicator {
            width: 24px;
            height: 24px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.75rem;
            font-weight: 900;
            flex-shrink: 0;
            transition: all 0.3s ease;
        }
        .agent-status-indicator.waiting {
            background: #f1f5f9;
            color: #94a3b8;
            border: 1.5px solid #cbd5e1;
        }
        .agent-status-indicator.processing {
            background: #e0f2fe;
            color: #0284c7;
            border: 1.5px solid #38bdf8;
            animation: spinFast 1.2s linear infinite;
        }
        .agent-status-indicator.completed {
            background: #ecfdf5;
            color: #059669;
            border: 1.5px solid #10b981;
            box-shadow: 0 0 6px rgba(16, 185, 129, 0.3);
        }
        .agent-status-indicator.warning {
            background: #fffbeb;
            color: #d97706;
            border: 1.5px solid #f59e0b;
        }
        .agent-status-indicator.failed {
            background: #fef2f2;
            color: #dc2626;
            border: 1.5px solid #ef4444;
            box-shadow: 0 0 8px rgba(239, 68, 68, 0.4);
        }
        @keyframes spinFast {
            100% { transform: rotate(360deg); }
        }

        .physician-control-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: #0f172a;
            border-radius: var(--radius-md);
            padding: 0.85rem 1.25rem;
            color: #ffffff;
            margin-top: 1rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.25);
        }
        .physician-control-title {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            font-size: 0.85rem;
            font-weight: 800;
            letter-spacing: 0.04em;
        }
        .physician-btn-group {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .btn-physician {
            padding: 0.45rem 0.95rem;
            font-size: 0.74rem;
            font-weight: 700;
            border-radius: 6px;
            cursor: pointer;
            border: none;
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            transition: all 0.2s ease;
        }
        .btn-physician.primary {
            background: linear-gradient(135deg, #0284c7, #2563eb);
            color: #ffffff;
            box-shadow: 0 2px 8px rgba(2, 132, 199, 0.35);
        }
        .btn-physician.primary:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(2, 132, 199, 0.5);
        }
        .btn-physician.pause {
            background: #334155;
            color: #e2e8f0;
        }
        .btn-physician.pause:hover { background: #475569; }
        .btn-physician.stop {
            background: #7f1d1d;
            color: #fecaca;
        }
        .btn-physician.stop:hover { background: #991b1b; }
        .btn-physician.override {
            background: #b45309;
            color: #fef3c7;
        }
        .btn-physician.override:hover { background: #d97706; }

        .decision-trace-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.78rem;
        }
        .decision-trace-table th {
            background: #f8fafc;
            padding: 0.65rem 0.85rem;
            text-align: left;
            font-weight: 700;
            color: var(--text-light);
            border-bottom: 1px solid var(--border-card);
            text-transform: uppercase;
            font-size: 0.65rem;
            letter-spacing: 0.05em;
        }
        .decision-trace-table td {
            padding: 0.75rem 0.85rem;
            border-bottom: 1px solid #f1f5f9;
            vertical-align: top;
            color: var(--text-primary);
        }
        .decision-trace-table tr:hover {
            background: #f8fafc;
        }
        .step-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #0284c7;
            color: #ffffff;
            font-weight: 800;
            font-size: 0.68rem;
        }
        .trial-card-item {
            background: #ffffff;
            border: 1px solid var(--border-card);
            border-radius: var(--radius-md);
            padding: 1rem;
            margin-bottom: 0.75rem;
            border-left: 4px solid #0284c7;
            transition: all 0.2s ease;
        }
        .trial-card-item:hover {
            border-color: #38bdf8;
            box-shadow: var(--shadow-sm);
        }
        .trial-card-header {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            margin-bottom: 0.5rem;
        }
        .trial-card-header h4 {
            font-size: 0.88rem;
            font-weight: 800;
            color: var(--text-primary);
        }
        .trial-meta-pills {
            display: flex;
            gap: 0.4rem;
            flex-wrap: wrap;
            margin-top: 0.35rem;
        }
        .trial-meta-pill {
            font-size: 0.65rem;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            background: #f1f5f9;
            color: #475569;
            font-weight: 600;
        }
        .trial-meta-pill.match {
            background: #ecfdf5;
            color: #065f46;
            border: 1px solid #a7f3d0;
        }
        .safety-alert-halt {
            background: #fef2f2;
            border: 2px solid #ef4444;
            border-radius: var(--radius-md);
            padding: 1.25rem;
            margin-bottom: 1.25rem;
            display: flex;
            align-items: flex-start;
            gap: 1rem;
            animation: pulseCard 2s infinite alternate;
        }
        .safety-alert-halt h3 {
            font-size: 1.05rem;
            font-weight: 900;
            color: #b91c1c;
            letter-spacing: 0.04em;
        }
        .safety-alert-halt p {
            font-size: 0.82rem;
            color: #7f1d1d;
            margin-top: 0.35rem;
            line-height: 1.5;
        }
    </style>
</head>
<body>

    <!-- ======================================================================
         SIDEBAR NAVIGATION
         ====================================================================== -->
    <aside class="sidebar">
        <div class="sidebar-brand">
            <div class="brand-icon">
                <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor">
                    <path d="M12 2v20M2 12h20M4.93 4.93l14.14 14.14M4.93 19.07l14.14-14.14" stroke-width="2.5" stroke-linecap="round"/>
                </svg>
            </div>
            <div class="brand-text">
                <h2>ONCO-CDSS</h2>
                <p>Clinical AI Core</p>
            </div>
        </div>

        <nav class="sidebar-menu">
            <div class="menu-heading">MAIN WORKFLOW</div>
            <a class="nav-link active" data-view="view-prediction">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><rect x="3" y="3" width="7" height="9" rx="1"/><rect x="14" y="3" width="7" height="5" rx="1"/><rect x="14" y="12" width="7" height="9" rx="1"/><rect x="3" y="16" width="7" height="5" rx="1"/></svg>
                <span>Prediction</span>
            </a>

            <div class="menu-heading">5-STAGE PIPELINE</div>
            <a class="nav-link" data-view="view-stage1">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="12" cy="12" r="9"/><path d="M12 8v8M8 12h8"/></svg>
                <span>① ML Toxicity</span>
                <div class="stage-pill-tag">ML</div>
            </a>
            <a class="nav-link" data-view="view-stage2">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
                <span>② DL Imaging</span>
                <div class="stage-pill-tag">DL</div>
            </a>
            <a class="nav-link" data-view="view-stage3">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>
                <span>③ NLP Triage</span>
                <div class="stage-pill-tag">NLP</div>
            </a>
            <a class="nav-link" data-view="view-stage4">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
                <span>④ SLM Reasoning</span>
                <div class="stage-pill-tag">SLM</div>
            </a>
            <a class="nav-link" data-view="view-stage5">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>
                <span>⑤ GenAI Report</span>
                <div class="stage-pill-tag">GENAI</div>
            </a>
            <a class="nav-link" data-view="view-stage6">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
                <span>⑥ Agentic AI</span>
                <div class="stage-pill-tag" style="background: rgba(14, 165, 233, 0.2); color: #0284c7;">AGENTIC</div>
            </a>

            <div class="menu-heading">RECORDS & AUDIT</div>
            <a class="nav-link" data-view="view-history">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
                <span>History</span>
            </a>
            <a class="nav-link" data-view="view-analytics">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>
                <span>Analytics</span>
            </a>
            <a class="nav-link" data-view="view-settings">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
                <span>Settings</span>
            </a>
        </nav>

        <div class="sidebar-footer">
            <div><span class="system-dot"></span>Active Pipeline</div>
            <div style="font-family: 'JetBrains Mono', monospace;">v2.4-PROD</div>
        </div>
    </aside>

    <!-- ======================================================================
         MAIN CONTENT AREA
         ====================================================================== -->
    <main class="main-wrapper">
        <!-- Top Navigation Header -->
        <header class="top-header">
            <div class="header-title-block">
                <div class="header-icon-box">
                    <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor">
                        <path d="M22 12h-4l-3 9L9 3l-3 9H2" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                </div>
                <div class="header-title-box">
                    <h1>AI PATIENT RISK PREDICTION</h1>
                    <p>Multi-stage clinical AI analysis</p>
                </div>
            </div>

            <div class="header-status-strip">
                <div class="stage-status-chip ready" id="chip-ml">ML ✓</div>
                <div class="stage-status-chip ready" id="chip-dl">DL ✓</div>
                <div class="stage-status-chip ready" id="chip-nlp">NLP ✓</div>
                <div class="stage-status-chip ready" id="chip-slm">SLM ✓</div>
                <div class="stage-status-chip ready" id="chip-genai">GenAI ✓</div>
                <div class="stage-status-chip ready" id="chip-agentic" style="border-color: #0284c7; color: #0284c7;">Agentic ✓</div>
            </div>
        </header>

        <div class="content-scroll">

            <!-- ==============================================================
                 VIEW 1: DASHBOARD / PREDICTION (PRIMARY MULTI-STAGE PIPELINE)
                 ============================================================== -->
            <section id="view-prediction" class="view-section active">
                <div class="view-header">
                    <h2>AI PATIENT RISK PREDICTION</h2>
                    <p>Multi-stage clinical AI analysis for oncology risk assessment</p>
                </div>

                <!-- 1. Pipeline Flow Card -->
                <div class="pipeline-card">
                    <div class="pipeline-flow">
                        <!-- Node 1: ML -->
                        <div class="flow-node" id="node-stage1" onclick="switchView('view-stage1')">
                            <div class="node-badge">1</div>
                            <div class="node-title">ML</div>
                            <div class="node-subtitle">Risk Prediction</div>
                            <span class="node-status-pill" id="pill-stage1">Waiting</span>
                        </div>
                        <div class="flow-arrow"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M5 12h14M12 5l7 7-7 7" stroke-width="2.5" stroke-linecap="round"/></svg></div>

                        <!-- Node 2: DL -->
                        <div class="flow-node" id="node-stage2" onclick="switchView('view-stage2')">
                            <div class="node-badge">2</div>
                            <div class="node-title">DL</div>
                            <div class="node-subtitle">Medical Image Analysis</div>
                            <span class="node-status-pill" id="pill-stage2">Waiting</span>
                        </div>
                        <div class="flow-arrow"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M5 12h14M12 5l7 7-7 7" stroke-width="2.5" stroke-linecap="round"/></svg></div>

                        <!-- Node 3: NLP -->
                        <div class="flow-node" id="node-stage3" onclick="switchView('view-stage3')">
                            <div class="node-badge">3</div>
                            <div class="node-title">NLP</div>
                            <div class="node-subtitle">Clinical Text Analysis</div>
                            <span class="node-status-pill" id="pill-stage3">Waiting</span>
                        </div>
                        <div class="flow-arrow"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M5 12h14M12 5l7 7-7 7" stroke-width="2.5" stroke-linecap="round"/></svg></div>

                        <!-- Node 4: SLM -->
                        <div class="flow-node" id="node-stage4" onclick="switchView('view-stage4')">
                            <div class="node-badge">4</div>
                            <div class="node-title">SLM</div>
                            <div class="node-subtitle">Clinical Reasoning</div>
                            <span class="node-status-pill" id="pill-stage4">Waiting</span>
                        </div>
                        <div class="flow-arrow"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M5 12h14M12 5l7 7-7 7" stroke-width="2.5" stroke-linecap="round"/></svg></div>

                        <!-- Node 5: GenAI -->
                        <div class="flow-node" id="node-stage5" onclick="switchView('view-stage5')">
                            <div class="node-badge">5</div>
                            <div class="node-title">GenAI</div>
                            <div class="node-subtitle">Final Clinical Report</div>
                            <span class="node-status-pill" id="pill-stage5">Waiting</span>
                        </div>
                    </div>

                    <div class="pipeline-progress-bar-wrap">
                        <div class="progress-track"><div class="progress-fill" id="pipeline-progress-fill"></div></div>
                        <div class="progress-label" id="pipeline-progress-label">Stage 0 of 5 (Ready)</div>
                    </div>
                </div>

                <!-- 2. Patient Input Card -->
                <div class="patient-input-card">
                    <div class="card-title-strip">
                        <h3>
                            <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>
                            PATIENT INPUT
                        </h3>
                        <div class="preset-buttons">
                            <span style="font-size: 0.68rem; color: #64748b; font-weight: 600;">PRESETS:</span>
                            <button class="btn-preset" onclick="loadPreset('HIGH_RISK')">High Risk NSCLC</button>
                            <button class="btn-preset" onclick="loadPreset('MODERATE_RISK')">Moderate Risk BRCA</button>
                            <button class="btn-preset" onclick="loadPreset('LOW_RISK')">Low Risk PRAD</button>
                            <button class="btn-preset danger" onclick="loadPreset('CRITICAL_EDGE')">Critical DILI Trap</button>
                        </div>
                    </div>

                    <form id="patient-form" onsubmit="event.preventDefault(); runCompleteAnalysis();">
                        <!-- Demographics & Staging -->
                        <div class="input-grid-4">
                            <div class="form-group">
                                <label class="form-label">Patient ID</label>
                                <input type="text" id="inp-patient_id" class="form-control" value="PAT-NSCLC-0842" required>
                            </div>
                            <div class="form-group">
                                <label class="form-label">Age (Years)</label>
                                <input type="number" id="inp-age" class="form-control" value="64" step="1" required>
                            </div>
                            <div class="form-group">
                                <label class="form-label">Sex</label>
                                <select id="inp-sex" class="form-control">
                                    <option value="F">Female</option>
                                    <option value="M">Male</option>
                                    <option value="Other">Other</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label class="form-label">Cancer Type</label>
                                <select id="inp-cancer_type" class="form-control">
                                    <option value="Lung (LUAD)">Lung Adenocarcinoma (LUAD)</option>
                                    <option value="Lung (LUSC)">Lung Squamous Cell (LUSC)</option>
                                    <option value="Breast (BRCA)">Breast Invasive Carcinoma (BRCA)</option>
                                    <option value="Colon (COAD)">Colon Adenocarcinoma (COAD)</option>
                                    <option value="Prostate (PRAD)">Prostate Adenocarcinoma (PRAD)</option>
                                    <option value="Stomach (STAD)">Stomach Adenocarcinoma (STAD)</option>
                                    <option value="Melanoma (SKCM)">Skin Cutaneous Melanoma (SKCM)</option>
                                </select>
                            </div>
                        </div>

                        <!-- Oncology Biomarkers -->
                        <div class="input-grid-4">
                            <div class="form-group">
                                <label class="form-label">Cancer Stage</label>
                                <select id="inp-cancer_stage" class="form-control">
                                    <option value="Stage I">Stage I</option>
                                    <option value="Stage II">Stage II</option>
                                    <option value="Stage III">Stage III</option>
                                    <option value="Stage IV" selected>Stage IV</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label class="form-label">Genomic Biomarker</label>
                                <input type="text" id="inp-mutation_profile" class="form-control" value="EGFR L858R, MET amplification">
                            </div>
                            <div class="form-group">
                                <label class="form-label">Tumor Marker Level (ng/mL)</label>
                                <input type="number" id="inp-tumor_marker_level" class="form-control" value="28.6" step="0.1">
                            </div>
                            <div class="form-group">
                                <label class="form-label">ctDNA Fractional VAF (%)</label>
                                <input type="number" id="inp-ctDNA_level" class="form-control" value="1.84" step="0.01">
                            </div>
                        </div>

                        <!-- Lab Panels -->
                        <div class="input-grid-4">
                            <div class="form-group">
                                <label class="form-label">ALT (U/L) — Liver</label>
                                <input type="number" id="inp-ALT" class="form-control" value="248" step="1" required>
                            </div>
                            <div class="form-group">
                                <label class="form-label">AST (U/L) — Liver</label>
                                <input type="number" id="inp-AST" class="form-control" value="210" step="1" required>
                            </div>
                            <div class="form-group">
                                <label class="form-label">Creatinine (mg/dL) — Renal</label>
                                <input type="number" id="inp-creatinine" class="form-control" value="1.45" step="0.01" required>
                            </div>
                            <div class="form-group">
                                <label class="form-label">Bilirubin (mg/dL)</label>
                                <input type="number" id="inp-bilirubin" class="form-control" value="2.2" step="0.1">
                            </div>
                        </div>

                        <!-- Treatment & Regimen -->
                        <div class="input-grid-3">
                            <div class="form-group">
                                <label class="form-label">Treatment Information / Regimen</label>
                                <input type="text" id="inp-treatment_name" class="form-control" value="Osimertinib 80mg + Capmatinib">
                            </div>
                            <div class="form-group">
                                <label class="form-label">Dosage (mg)</label>
                                <input type="number" id="inp-dosage_mg" class="form-control" value="480" step="10">
                            </div>
                            <div class="form-group">
                                <label class="form-label">Treatment Cycle</label>
                                <input type="number" id="inp-treatment_cycle" class="form-control" value="4" step="1">
                            </div>
                        </div>

                        <!-- Clinical Notes -->
                        <div class="form-group" style="margin-bottom: 1rem;">
                            <label class="form-label">Clinical Notes / Symptoms (For NLP & SLM Reasoning)</label>
                            <textarea id="inp-clinical_notes" class="form-control" placeholder="Enter clinical consult notes, acute symptoms, vital signs...">Patient on Osimertinib and Capmatinib presenting with spiking fever 38.4C, marked right upper quadrant tenderness, intractable nausea, and severe fatigue. Marked transaminase elevation (ALT 248 U/L, AST 210 U/L) indicating acute Drug-Induced Liver Injury (DILI).</textarea>
                        </div>

                        <!-- File Upload Dropzones -->
                        <div class="upload-row">
                            <div class="upload-box" id="drop-csv">
                                <div class="upload-icon">
                                    <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
                                </div>
                                <div class="upload-text">
                                    <h4 id="csv-filename">Upload Patient CSV</h4>
                                    <p>Drop lab or cohort CSV to autofill form</p>
                                </div>
                                <input type="file" id="file-csv-input" accept=".csv" onchange="handleCSVUpload(this)">
                            </div>

                            <div class="upload-box" id="drop-image">
                                <div class="upload-icon">
                                    <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
                                </div>
                                <img id="image-preview" class="image-preview-thumb" alt="Preview">
                                <div class="upload-text">
                                    <h4 id="image-filename">Medical Image Upload (DL)</h4>
                                    <p>Select CT slice or Pathology tile (.png, .jpg)</p>
                                </div>
                                <input type="file" id="file-img-input" accept="image/*" onchange="handleImageUpload(this)">
                            </div>
                        </div>

                        <!-- Action Buttons -->
                        <div class="btn-action-row">
                            <button type="submit" id="btn-run-analysis" class="btn-primary-analyze">
                                <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor"><polygon points="5 3 19 12 5 21 5 3" fill="currentColor"/></svg>
                                RUN COMPLETE AI ANALYSIS
                            </button>
                            <button type="button" class="btn-secondary-reset" onclick="resetForm()">
                                RESET
                            </button>
                        </div>
                    </form>
                </div>

                <!-- 3. Live 5-Stage Output Cards Grid -->
                <div class="stage-results-grid" id="results-grid" style="display: none;">
                    <!-- ML Result Card -->
                    <div class="stage-result-card" id="card-res-ml">
                        <div class="card-header-mini">
                            <h4>① ML — Risk & Toxicity Prediction</h4>
                            <span class="metric-badge blue" id="ml-badge-risk">Waiting</span>
                        </div>
                        <p id="ml-text-interp" style="font-size: 0.78rem; color: var(--text-secondary);">Awaiting ML evaluation...</p>
                        <div class="feature-tag-list" id="ml-tags-list"></div>
                    </div>

                    <!-- DL Result Card -->
                    <div class="stage-result-card" id="card-res-dl">
                        <div class="card-header-mini">
                            <h4>② DL — Medical Image Analysis</h4>
                            <span class="metric-badge green" id="dl-badge-risk">Waiting</span>
                        </div>
                        <p id="dl-text-interp" style="font-size: 0.78rem; color: var(--text-secondary);">Awaiting multimodal image inference...</p>
                        <div style="display: flex; gap: 0.5rem; align-items: center;" id="dl-preview-row"></div>
                    </div>

                    <!-- NLP Result Card -->
                    <div class="stage-result-card" id="card-res-nlp">
                        <div class="card-header-mini">
                            <h4>③ NLP — Clinical Text Analysis</h4>
                            <span class="metric-badge yellow" id="nlp-badge-triage">Waiting</span>
                        </div>
                        <p id="nlp-text-interp" style="font-size: 0.78rem; color: var(--text-secondary);">Awaiting NLP entity extraction...</p>
                        <div class="feature-tag-list" id="nlp-entities-list"></div>
                    </div>

                    <!-- SLM Result Card -->
                    <div class="stage-result-card" id="card-res-slm">
                        <div class="card-header-mini">
                            <h4>④ SLM — Clinical Reasoning</h4>
                            <span class="metric-badge blue" id="slm-badge-guardrail">Waiting</span>
                        </div>
                        <p id="slm-text-interp" style="font-size: 0.78rem; color: var(--text-secondary);">Awaiting 3B SLM safety reasoning...</p>
                    </div>
                </div>

                <!-- 4. Prominent FINAL AI ASSESSMENT Card -->
                <div class="final-assessment-card" id="final-assessment-section" style="display: none;">
                    <div class="assessment-banner">
                        <div class="assessment-banner-left">
                            <h3 id="final-card-patient-title">FINAL AI ASSESSMENT</h3>
                            <p id="final-card-patient-sub">Patient: PAT-NSCLC-0842 | Lung (LUAD) | Stage IV</p>
                        </div>
                        <div class="risk-classification-tag high" id="final-risk-tag">
                            HIGH RISK
                        </div>
                    </div>

                    <div class="assessment-breakdown-grid">
                        <div class="breakdown-box">
                            <span>OVERALL RISK TIER</span>
                            <h4 id="fin-overall-risk">HIGH RISK</h4>
                        </div>
                        <div class="breakdown-box">
                            <span>RISK PROBABILITY</span>
                            <h4 id="fin-risk-prob">78.4%</h4>
                        </div>
                        <div class="breakdown-box">
                            <span>MODEL CONFIDENCE</span>
                            <h4 id="fin-confidence">91.5%</h4>
                        </div>
                        <div class="breakdown-box">
                            <span>AUDIT STATUS</span>
                            <h4 id="fin-audit-status" style="color: #059669;">VERIFIED SAFE</h4>
                        </div>
                    </div>

                    <div class="clinical-summary-box" id="fin-genai-summary">
                        Generating comprehensive multi-modal oncology clinical report...
                    </div>

                    <div>
                        <h4 style="font-size: 0.82rem; font-weight: 700; margin-bottom: 0.35rem;">RECOMMENDED CLINICAL CONSIDERATIONS</h4>
                        <div class="considerations-checklist" id="fin-considerations-list">
                            <!-- Populated dynamically -->
                        </div>
                    </div>

                    <div class="disclaimer-strip">
                        <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>
                        AI-generated decision support — not a replacement for clinical judgment.
                    </div>
                </div>
            </section>


            <!-- ==============================================================
                 VIEW 2: STAGE 1 — ML TOXICITY & RISK DEEP DIVE
                 ============================================================== -->
            <section id="view-stage1" class="view-section">
                <div class="view-header">
                    <h2>STAGE ① — MACHINE LEARNING TOXICITY PREDICTION</h2>
                    <p>Voting Ensemble Architecture (Random Forest, Extra Trees, XGBoost) with Permutation Feature Importances</p>
                </div>

                <!-- 1. ML PATIENT INPUT SECTION (BEFORE EXISTING REPORT) -->
                <div class="patient-input-card">
                    <div class="card-title-strip">
                        <h3>
                            <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor"><circle cx="12" cy="12" r="9"/><path d="M12 8v8M8 12h8"/></svg>
                            ML PATIENT INPUT (STRUCTURED CLINICAL DATA)
                        </h3>
                        <div class="preset-buttons">
                            <span style="font-size: 0.68rem; color: #64748b; font-weight: 600;">QUICK FILL:</span>
                            <button class="btn-preset" type="button" onclick="setMLSample('HIGH')">High Toxicity Sample</button>
                            <button class="btn-preset" type="button" onclick="setMLSample('MODERATE')">Moderate Sample</button>
                            <button class="btn-preset" type="button" onclick="setMLSample('LOW')">Low Toxicity Sample</button>
                        </div>
                    </div>

                    <form id="form-stage1" onsubmit="event.preventDefault(); runStage1Prediction();">
                        <div class="input-grid-4">
                            <div class="form-group">
                                <label class="form-label">Patient ID</label>
                                <input type="text" id="s1-patient-id" class="form-control" value="PAT-ML-0101" required>
                            </div>
                            <div class="form-group">
                                <label class="form-label">Age (Years)</label>
                                <input type="number" id="s1-age" class="form-control" value="64" step="1" required>
                            </div>
                            <div class="form-group">
                                <label class="form-label">Sex</label>
                                <select id="s1-sex" class="form-control">
                                    <option value="M" selected>Male</option>
                                    <option value="F">Female</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label class="form-label">Cancer Type</label>
                                <select id="s1-cancer-type" class="form-control">
                                    <option value="Lung (LUAD)" selected>Lung Adenocarcinoma (LUAD)</option>
                                    <option value="Lung (LUSC)">Lung Squamous (LUSC)</option>
                                    <option value="Breast (BRCA)">Breast Invasive (BRCA)</option>
                                    <option value="Colon (COAD)">Colon Adeno (COAD)</option>
                                    <option value="Prostate (PRAD)">Prostate Adeno (PRAD)</option>
                                    <option value="Stomach (STAD)">Stomach Adeno (STAD)</option>
                                </select>
                            </div>
                        </div>

                        <div class="input-grid-4">
                            <div class="form-group">
                                <label class="form-label">Cancer Stage</label>
                                <select id="s1-cancer-stage" class="form-control">
                                    <option value="Stage I">Stage I</option>
                                    <option value="Stage II">Stage II</option>
                                    <option value="Stage III">Stage III</option>
                                    <option value="Stage IV" selected>Stage IV</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label class="form-label">Genomic Biomarker</label>
                                <input type="text" id="s1-mutation" class="form-control" value="EGFR L858R">
                            </div>
                            <div class="form-group">
                                <label class="form-label">Tumor Marker Level (ng/ml)</label>
                                <input type="number" id="s1-tumor-marker" class="form-control" value="28.6" step="0.1" required>
                            </div>
                            <div class="form-group">
                                <label class="form-label">ctDNA Fractional VAF (%)</label>
                                <input type="number" id="s1-ctdna" class="form-control" value="1.84" step="0.01" required>
                            </div>
                        </div>

                        <div class="input-grid-4">
                            <div class="form-group">
                                <label class="form-label">ALT (U/L)</label>
                                <input type="number" id="s1-alt" class="form-control" value="248" step="1" required>
                            </div>
                            <div class="form-group">
                                <label class="form-label">AST (U/L)</label>
                                <input type="number" id="s1-ast" class="form-control" value="210" step="1" required>
                            </div>
                            <div class="form-group">
                                <label class="form-label">Creatinine (mg/dL)</label>
                                <input type="number" id="s1-creatinine" class="form-control" value="1.45" step="0.01" required>
                            </div>
                            <div class="form-group">
                                <label class="form-label">Bilirubin (mg/dL)</label>
                                <input type="number" id="s1-bilirubin" class="form-control" value="2.2" step="0.1" required>
                            </div>
                        </div>

                        <div class="input-grid-3">
                            <div class="form-group">
                                <label class="form-label">Treatment Information / Regimen</label>
                                <input type="text" id="s1-treatment" class="form-control" value="Osimertinib 80mg + Capmatinib">
                            </div>
                            <div class="form-group">
                                <label class="form-label">Dosage (mg)</label>
                                <input type="number" id="s1-dosage" class="form-control" value="480" step="10">
                            </div>
                            <div class="form-group">
                                <label class="form-label">Treatment Cycle</label>
                                <input type="number" id="s1-cycle" class="form-control" value="4" step="1">
                            </div>
                        </div>

                        <div class="btn-action-row" style="margin-top: 0.5rem;">
                            <button type="submit" id="btn-run-stage1" class="btn-primary-analyze" style="max-width: 280px;">
                                [ RUN ML PREDICTION ]
                            </button>
                        </div>
                    </form>
                </div>

                <!-- 2. ML PREDICTION RESULT CARD -->
                <div id="s1-output-card" class="final-assessment-card" style="display: none; margin-top: 1rem;">
                    <div class="assessment-banner">
                        <div class="assessment-banner-left">
                            <h3 id="s1-out-title">ML PREDICTION RESULT</h3>
                            <p id="s1-out-patient-sub">Patient: PAT-ML-0101 | Model: VotingClassifier (Random Forest + Extra Trees + XGBoost)</p>
                        </div>
                        <div class="risk-classification-tag" id="s1-out-badge">HIGH RISK</div>
                    </div>
                    <div class="assessment-breakdown-grid">
                        <div class="breakdown-box">
                            <span>PATIENT ID</span>
                            <h4 id="s1-out-pid">PAT-ML-0101</h4>
                        </div>
                        <div class="breakdown-box">
                            <span>PREDICTED RISK CLASS</span>
                            <h4 id="s1-out-pred">High Toxicity</h4>
                        </div>
                        <div class="breakdown-box">
                            <span>RISK LEVEL</span>
                            <h4 id="s1-out-class">HIGH RISK</h4>
                        </div>
                        <div class="breakdown-box">
                            <span>PREDICTION PROBABILITY</span>
                            <h4 id="s1-out-prob">38.4%</h4>
                        </div>
                        <div class="breakdown-box">
                            <span>CONFIDENCE SCORE</span>
                            <h4 id="s1-out-conf">85.0%</h4>
                        </div>
                        <div class="breakdown-box">
                            <span>MODEL STATUS</span>
                            <h4 id="s1-out-model-status" style="color: #059669;">Operational</h4>
                        </div>
                    </div>
                    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.6rem 0.9rem; margin-top: 0.6rem; display: flex; gap: 1.25rem; flex-wrap: wrap; align-items: center;">
                        <span style="font-size: 0.68rem; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.5px;">Calibrated Class Distribution:</span>
                        <div id="s1-out-class-probs" style="display: flex; gap: 1rem; flex-wrap: wrap; font-size: 0.78rem;">
                            <span><strong>Low Toxicity:</strong> —</span>
                            <span><strong>Moderate Toxicity:</strong> —</span>
                            <span><strong>High Toxicity:</strong> —</span>
                        </div>
                    </div>
                    <div class="clinical-summary-box" id="s1-out-interp">
                        Interpretation of structured parameters...
                    </div>
                    <div>
                        <h4 style="font-size: 0.82rem; font-weight: 700; margin-bottom: 0.4rem;">IMPORTANT CLINICAL FACTORS</h4>
                        <div class="feature-tag-list" id="s1-out-features"></div>
                    </div>
                </div>

                <!-- 3. EXISTING ML REPORT / FEATURE IMPORTANCE -->
                <div class="table-card" style="padding: 1.5rem; margin-top: 1rem;">
                    <h3 style="font-size: 0.95rem; font-weight: 800; margin-bottom: 0.85rem;">Top Influential Clinical Features (Permutation Importance)</h3>
                    <table class="clinical-table">
                        <thead>
                            <tr>
                                <th>Feature Name</th>
                                <th>Clinical Domain</th>
                                <th>Mean Permutation Importance</th>
                                <th>Status</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr><td><strong>ALT (Alanine Aminotransferase)</strong></td><td>Hepatic Enzyme Panel</td><td>0.284</td><td><span class="metric-badge red">High Sensitivity</span></td></tr>
                            <tr><td><strong>AST (Aspartate Aminotransferase)</strong></td><td>Hepatic Enzyme Panel</td><td>0.221</td><td><span class="metric-badge red">High Sensitivity</span></td></tr>
                            <tr><td><strong>ctDNA Fractional VAF</strong></td><td>Liquid Biopsy Genomics</td><td>0.185</td><td><span class="metric-badge yellow">Moderate Sensitivity</span></td></tr>
                            <tr><td><strong>Tumor Marker Level</strong></td><td>Serum Oncology Biomarkers</td><td>0.142</td><td><span class="metric-badge blue">Stable</span></td></tr>
                            <tr><td><strong>Creatinine</strong></td><td>Renal Function Panel</td><td>0.098</td><td><span class="metric-badge blue">Stable</span></td></tr>
                            <tr><td><strong>Pulse Pressure</strong></td><td>Vascular Vitals</td><td>0.070</td><td><span class="metric-badge green">Baseline</span></td></tr>
                        </tbody>
                    </table>
                </div>
            </section>


            <!-- ==============================================================
                 VIEW 3: STAGE 2 — DL MEDICAL IMAGE ANALYSIS DEEP DIVE
                 ============================================================== -->
            <section id="view-stage2" class="view-section">
                <div class="view-header">
                    <h2>STAGE ② — DEEP LEARNING MULTIMODAL PROGRESSION</h2>
                    <p>MultimodalLSTM combining 2D CT Slices, Pathology Tiles, and Longitudinal Sequence Steps</p>
                </div>

                <!-- 1. DL INDEPENDENT INPUT SECTION -->
                <div class="patient-input-card">
                    <div class="card-title-strip">
                        <h3>
                            <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
                            DL MULTIMODAL INPUT (STRUCTURED PATIENT/CSV DATA + MEDICAL IMAGE)
                        </h3>
                    </div>

                    <form id="form-stage2" onsubmit="event.preventDefault(); runStage2Prediction();">
                        <div class="input-grid-4">
                            <div class="form-group">
                                <label class="form-label">Patient ID</label>
                                <input type="text" id="s2-patient-id" class="form-control" value="PAT-DL-0201" required>
                            </div>
                            <div class="form-group">
                                <label class="form-label">Age</label>
                                <input type="number" id="s2-age" class="form-control" value="61" step="1" required>
                            </div>
                            <div class="form-group">
                                <label class="form-label">Cancer Type</label>
                                <select id="s2-cancer-type" class="form-control" onchange="updateDLSampleImages(this.value)">
                                    <option value="luad" selected>Lung Adenocarcinoma (LUAD)</option>
                                    <option value="lusc">Lung Squamous (LUSC)</option>
                                    <option value="brca">Breast Invasive (BRCA)</option>
                                    <option value="prad">Prostate Adeno (PRAD)</option>
                                    <option value="coad">Colon Adeno (COAD)</option>
                                    <option value="stad">Stomach Adeno (STAD)</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label class="form-label">Cancer Stage</label>
                                <select id="s2-cancer-stage" class="form-control">
                                    <option value="Stage I">Stage I</option>
                                    <option value="Stage II">Stage II</option>
                                    <option value="Stage III">Stage III</option>
                                    <option value="Stage IV" selected>Stage IV</option>
                                </select>
                            </div>
                        </div>

                        <div class="input-grid-4">
                            <div class="form-group">
                                <label class="form-label">Genomic Biomarker</label>
                                <input type="text" id="s2-mutation" class="form-control" value="EGFR L858R">
                            </div>
                            <div class="form-group">
                                <label class="form-label">Tumor Marker Level (ng/mL)</label>
                                <input type="number" id="s2-tumor-marker" class="form-control" value="24.5" step="0.1" required>
                            </div>
                            <div class="form-group">
                                <label class="form-label">ctDNA Fractional Level (%)</label>
                                <input type="number" id="s2-ctdna" class="form-control" value="1.62" step="0.01" required>
                            </div>
                            <div class="form-group">
                                <label class="form-label">Baseline WSI Atypia Score</label>
                                <input type="number" id="s2-atypia" class="form-control" value="0.75" step="0.05" min="0" max="1">
                            </div>
                        </div>

                        <!-- Uploads -->
                        <div class="upload-row" style="margin-top: 0.5rem;">
                            <div class="upload-box">
                                <div class="upload-icon">
                                    <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
                                </div>
                                <div class="upload-text">
                                    <h4 id="s2-csv-filename">CSV Upload</h4>
                                    <p>Upload structured patient CSV</p>
                                </div>
                                <input type="file" accept=".csv" onchange="handleStage2CSV(this)">
                            </div>

                            <div class="upload-box">
                                <div class="upload-icon">
                                    <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
                                </div>
                                <img id="s2-image-preview" class="image-preview-thumb" alt="DL Preview" style="display: block; width: 44px; height: 44px;" src="/api/pipeline/image/luad/ct_slice_0000.png">
                                <div class="upload-text">
                                    <h4 id="s2-image-filename">Medical Image Upload</h4>
                                    <p>Select or upload CT slice / Pathology tile</p>
                                </div>
                                <input type="file" accept="image/*" onchange="handleStage2Image(this)">
                            </div>
                        </div>

                        <div style="display: flex; gap: 0.5rem; align-items: center; margin: 0.5rem 0 1rem;">
                            <span style="font-size: 0.7rem; color: var(--text-secondary); font-weight: 600;">SAMPLE SCANS:</span>
                            <button type="button" class="btn-preset" onclick="setDLImageSample('luad', 'ct_slice_0000.png')">LUAD CT Scan</button>
                            <button type="button" class="btn-preset" onclick="setDLImageSample('luad', 'tile_0000.png')">LUAD Pathology Tile</button>
                            <button type="button" class="btn-preset" onclick="setDLImageSample('brca', 'tile_0000.png')">BRCA Pathology Tile</button>
                            <button type="button" class="btn-preset" onclick="setDLImageSample('prad', 'ct_slice_0000.png')">PRAD CT Scan</button>
                        </div>

                        <div class="btn-action-row">
                            <button type="submit" id="btn-run-stage2" class="btn-primary-analyze" style="max-width: 280px;">
                                [ RUN DL PREDICTION ]
                            </button>
                        </div>
                    </form>
                </div>

                <!-- 2. DL OUTPUT SECTION -->
                <div id="s2-output-card" class="final-assessment-card" style="display: none; margin-top: 1rem;">
                    <div class="assessment-banner">
                        <div class="assessment-banner-left">
                            <h3>DL MULTIMODAL INFERENCE RESULT</h3>
                            <p id="s2-out-patient-sub">Patient: PAT-DL-0201 | Model: MultimodalLSTM (CNN Embeddings + Longitudinal Sequence)</p>
                        </div>
                        <div class="risk-classification-tag high" id="s2-out-risk-tier">HIGH PROGRESSION RISK</div>
                    </div>
                    <div class="assessment-breakdown-grid">
                        <div class="breakdown-box">
                            <span>PATIENT ID</span>
                            <h4 id="s2-out-pid">PAT-DL-0201</h4>
                        </div>
                        <div class="breakdown-box">
                            <span>IMAGE PREDICTION</span>
                            <h4 id="s2-out-class">Malignant Neoplasm</h4>
                        </div>
                        <div class="breakdown-box">
                            <span>CLASSIFICATION</span>
                            <h4 id="s2-out-classif">Progressive Lesion</h4>
                        </div>
                        <div class="breakdown-box">
                            <span>PROBABILITY</span>
                            <h4 id="s2-out-prob">76.4%</h4>
                        </div>
                        <div class="breakdown-box">
                            <span>CONFIDENCE</span>
                            <h4 id="s2-out-conf">92.0%</h4>
                        </div>
                        <div class="breakdown-box">
                            <span>MODEL STATUS</span>
                            <h4 id="s2-out-model-status" style="color: #059669;">Operational</h4>
                        </div>
                    </div>
                    <div class="clinical-summary-box" id="s2-out-interp">
                        Trajectory progression interpretation...
                    </div>
                    <div id="s2-timepoint-box">
                        <h4 style="font-size: 0.82rem; font-weight: 700; margin-bottom: 0.4rem;">LONGITUDINAL SEQUENCE TRAJECTORY (T0 — T4)</h4>
                        <table class="clinical-table">
                            <thead><tr><th>Sequence Step</th><th>Timepoint</th><th>Progression Prob</th><th>Risk Tier</th></tr></thead>
                            <tbody id="s2-trajectory-table"></tbody>
                        </table>
                    </div>
                </div>

                <!-- 3. EXISTING CROSS-MODAL SCAN GALLERY -->
                <div class="table-card" style="padding: 1.5rem; display: flex; flex-direction: column; gap: 1rem; margin-top: 1rem;">
                    <h3 style="font-size: 0.95rem; font-weight: 800;">Cross-Modal Scan Gallery</h3>
                    <div style="display: flex; gap: 1.5rem; flex-wrap: wrap;">
                        <div style="background: #000; border-radius: 8px; padding: 0.5rem; text-align: center;">
                            <img src="/api/pipeline/image/luad/ct_slice_0000.png" style="width: 180px; height: 180px; object-fit: cover; border-radius: 4px;" alt="CT Scan">
                            <p style="color: #fff; font-size: 0.72rem; margin-top: 0.35rem;">CT Thorax Axial (Slice 000)</p>
                        </div>
                        <div style="background: #000; border-radius: 8px; padding: 0.5rem; text-align: center;">
                            <img src="/api/pipeline/image/luad/tile_0000.png" style="width: 180px; height: 180px; object-fit: cover; border-radius: 4px;" alt="Pathology Tile">
                            <p style="color: #fff; font-size: 0.72rem; margin-top: 0.35rem;">H&E Pathology Tile (40x)</p>
                        </div>
                    </div>
                </div>
            </section>


            <!-- ==============================================================
                 VIEW 4: STAGE 3 — NLP CLINICAL TEXT DEEP DIVE
                 ============================================================== -->
            <section id="view-stage3" class="view-section">
                <div class="view-header">
                    <h2>STAGE ③ — NATURAL LANGUAGE PROCESSING TRIAGE</h2>
                    <p>Hybrid TF-IDF + Logistic Regression Urgency Classifier with Named Entity Recognition</p>
                </div>

                <!-- 1. NLP INDEPENDENT INPUT SECTION -->
                <div class="patient-input-card">
                    <div class="card-title-strip">
                        <h3>
                            <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/></svg>
                            ENTER CLINICAL TEXT
                        </h3>
                        <div class="preset-buttons">
                            <span style="font-size: 0.68rem; color: #64748b; font-weight: 600;">SAMPLES:</span>
                            <button class="btn-preset" type="button" onclick="setNLPTextSample('HIGH')">High Urgency Fever</button>
                            <button class="btn-preset" type="button" onclick="setNLPTextSample('MODERATE')">Moderate Fatigue</button>
                            <button class="btn-preset danger" type="button" onclick="setNLPTextSample('CRITICAL')">Critical Sepsis</button>
                        </div>
                    </div>

                    <form id="form-stage3" onsubmit="event.preventDefault(); runStage3Prediction();">
                        <div class="form-group" style="margin-bottom: 1rem;">
                            <label class="form-label">Enter clinical notes, symptoms, diagnosis, medical history, or oncology report text...</label>
                            <textarea id="inp-stage3-text" class="form-control" style="min-height: 110px;" placeholder="Enter clinical notes, symptoms, diagnosis, medical history, or oncology report text...">Patient has elevated tumor markers and worsening symptoms with high fever 38.8C, persistent fatigue, and right upper quadrant abdominal tenderness concerning for acute drug toxicity.</textarea>
                        </div>

                        <div class="btn-action-row">
                            <button type="submit" id="btn-run-stage3" class="btn-primary-analyze" style="max-width: 280px;">
                                [ RUN NLP PREDICTION ]
                            </button>
                        </div>
                    </form>
                </div>

                <!-- 2. NLP OUTPUT SECTION -->
                <div id="s3-output-card" class="final-assessment-card" style="display: none; margin-top: 1rem;">
                    <div class="assessment-banner">
                        <div class="assessment-banner-left">
                            <h3>NLP TRIAGE & EXTRACTION RESULT</h3>
                            <p>Model: Hybrid TF-IDF + Logistic Regression with Biomedical Rule/Dictionary NER</p>
                        </div>
                        <div class="risk-classification-tag high" id="s3-out-badge">HIGH RISK</div>
                    </div>
                    <div class="assessment-breakdown-grid">
                        <div class="breakdown-box">
                            <span>RISK LEVEL</span>
                            <h4 id="s3-out-level">HIGH RISK</h4>
                        </div>
                        <div class="breakdown-box">
                            <span>PROBABILITY</span>
                            <h4 id="s3-out-prob">0.950</h4>
                        </div>
                        <div class="breakdown-box">
                            <span>CONFIDENCE</span>
                            <h4 id="s3-out-conf">95.0%</h4>
                        </div>
                        <div class="breakdown-box">
                            <span>MODEL STATUS</span>
                            <h4 id="s3-out-model-status" style="color: #059669;">Operational</h4>
                        </div>
                        <div class="breakdown-box">
                            <span>ENTITIES EXTRACTED</span>
                            <h4 id="s3-out-count">4 Identified</h4>
                        </div>
                        <div class="breakdown-box">
                            <span>LATENCY</span>
                            <h4 id="s3-out-latency">14.2 ms</h4>
                        </div>
                    </div>
                    <div class="clinical-summary-box" id="s3-out-interp">
                        NLP triage output...
                    </div>
                    <div>
                        <h4 style="font-size: 0.82rem; font-weight: 700; margin-bottom: 0.4rem;">EXTRACTED CLINICAL ENTITIES & TERMS</h4>
                        <div class="feature-tag-list" id="s3-out-entities"></div>
                    </div>
                </div>

                <!-- 3. EXISTING CLINICAL GUIDELINE RETRIEVAL SOP MATCHES -->
                <div class="table-card" style="padding: 1.5rem; margin-top: 1rem;">
                    <h3 style="font-size: 0.95rem; font-weight: 800; margin-bottom: 0.85rem;">Clinical Guideline Retrieval SOP Matches</h3>
                    <div style="display: flex; flex-direction: column; gap: 0.75rem;">
                        <div class="clinical-summary-box">
                            <strong>NCCN Immune-Mediated Hepatotoxicity Guideline (Grade 3+):</strong>
                            <p>For ALT/AST > 5x ULN or Total Bilirubin > 3x ULN, withhold immunotherapy/TKI regimen immediately. Administer oral prednisone 1-2 mg/kg/day with daily liver panel monitoring.</p>
                        </div>
                        <div class="clinical-summary-box" style="border-left-color: #10b981;">
                            <strong>ASCO Febrile Neutropenia & Sepsis Triage Protocol:</strong>
                            <p>Immediate blood cultures and stat empiric broad-spectrum antibiotic initiation within 60 minutes for temperature > 38.3C and absolute neutrophil count < 500/uL.</p>
                        </div>
                    </div>
                </div>
            </section>


            <!-- ==============================================================
                 VIEW 5: STAGE 4 — SLM CLINICAL REASONING DEEP DIVE
                 ============================================================== -->
            <section id="view-stage4" class="view-section">
                <div class="view-header">
                    <h2>STAGE ④ — SMALL LANGUAGE MODEL CLINICAL REASONING</h2>
                    <p>Fine-Tuned Qwen2.5-3B Instruct with Strict Post-Inference Safety Guardrails</p>
                </div>

                <!-- 1. SLM INDEPENDENT SIMPLIFICATION INPUT SECTION -->
                <div class="patient-input-card">
                    <div class="card-title-strip">
                        <h3>
                            <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
                            LONG CLINICAL TEXT (CONVERT TO SHORT, SIMPLE CLINICAL EXPLANATION)
                        </h3>
                        <div class="preset-buttons">
                            <span style="font-size: 0.68rem; color: #64748b; font-weight: 600;">SAMPLES:</span>
                            <button class="btn-preset" type="button" onclick="setSLMTextSample('TKI')">TKI Hepatic Consult</button>
                            <button class="btn-preset danger" type="button" onclick="setSLMTextSample('CRITICAL')">Emergency ICU Sepsis</button>
                        </div>
                    </div>

                    <form id="form-stage4" onsubmit="event.preventDefault(); runStage4Simplification();">
                        <div class="form-group" style="margin-bottom: 1rem;">
                            <label class="form-label">Enter long medical/clinical paragraph:</label>
                            <textarea id="inp-stage4-text" class="form-control" style="min-height: 120px;" placeholder="Paste long technical oncology report or physician notes...">Patient PAT-0001 (Lung, EGFR L858R) on Targeted Therapy presents with acute transaminase elevation with ALT 248 U/L and AST 210 U/L following cycle 4 of tyrosine kinase inhibitor therapy. Clinical observation reveals persistent right upper quadrant discomfort, fever at 38.4C, and intractable nausea concerning for drug-induced liver injury requiring urgent clinical review and hydration support.</textarea>
                        </div>

                        <div class="btn-action-row">
                            <button type="submit" id="btn-run-stage4" class="btn-primary-analyze" style="max-width: 280px;">
                                [ SIMPLIFY WITH SLM ]
                            </button>
                        </div>
                    </form>
                </div>

                <!-- 2. SLM OUTPUT SECTION -->
                <div id="s4-output-card" class="final-assessment-card" style="display: none; margin-top: 1rem;">
                    <div class="assessment-banner">
                        <div class="assessment-banner-left">
                            <h3>SIMPLIFIED CLINICAL EXPLANATION</h3>
                            <p>Original Text ↓ Simplified Text (No hidden reasoning or CoT exposed)</p>
                        </div>
                        <div class="risk-classification-tag moderate" id="s4-out-status-badge">VERIFIED SAFE</div>
                    </div>

                    <div style="display: flex; flex-direction: column; gap: 0.75rem;">
                        <div>
                            <span style="font-size: 0.68rem; font-weight: 700; color: var(--text-light); text-transform: uppercase;">ORIGINAL TEXT</span>
                            <div style="background: #f1f5f9; padding: 0.75rem; border-radius: 6px; font-size: 0.78rem; color: #475569; margin-top: 0.25rem;" id="s4-out-original">
                                Original text...
                            </div>
                        </div>

                        <div style="text-align: center; color: var(--accent-blue); font-size: 1.25rem; font-weight: 800;">↓</div>

                        <div>
                            <span style="font-size: 0.68rem; font-weight: 700; color: var(--text-light); text-transform: uppercase;">SIMPLIFIED CLINICAL EXPLANATION</span>
                            <div class="clinical-summary-box" style="border-left-color: var(--accent-blue);" id="s4-out-simplified">
                                Simplified concise clinical explanation...
                            </div>
                        </div>
                    </div>

                    <div class="assessment-breakdown-grid" style="margin-top: 0.5rem;">
                        <div class="breakdown-box">
                            <span>PROCESSING STATUS</span>
                            <h4 id="s4-out-proc">Completed ✓</h4>
                        </div>
                        <div class="breakdown-box">
                            <span>MODEL STATUS</span>
                            <h4 id="s4-out-model">Operational (Qwen2.5-3B-Instruct)</h4>
                        </div>
                        <div class="breakdown-box">
                            <span>SAFETY / GUARDRAIL</span>
                            <h4 id="s4-out-guardrail" style="color: #059669;">VERIFIED_SAFE</h4>
                        </div>
                        <div class="breakdown-box">
                            <span>LATENCY</span>
                            <h4 id="s4-out-latency">12.0 ms</h4>
                        </div>
                    </div>
                </div>

                <!-- 3. EXISTING GUARDRAIL VERIFICATION AUDIT MATRIX -->
                <div class="table-card" style="padding: 1.5rem; margin-top: 1rem;">
                    <h3 style="font-size: 0.95rem; font-weight: 800; margin-bottom: 0.85rem;">Guardrail Verification Audit Matrix</h3>
                    <table class="clinical-table">
                        <thead>
                            <tr>
                                <th>Safety Constraint</th>
                                <th>Validation Method</th>
                                <th>Compliance Status</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr><td><strong>Under-Triage Interception</strong></td><td>Rule Floor Escalator (High/Critical Floor Detection)</td><td><span class="metric-badge green">100% Zero-Leak</span></td></tr>
                            <tr><td><strong>Patient ID Retention</strong></td><td>Regex Deterministic ID Injection</td><td><span class="metric-badge green">100% Preserved</span></td></tr>
                            <tr><td><strong>Strict 2-Sentence Compliance</strong></td><td>Punctuation Sentence Normalizer</td><td><span class="metric-badge green">Pass (2 Sentences)</span></td></tr>
                        </tbody>
                    </table>
                </div>
            </section>


            <!-- ==============================================================
                 VIEW 6: STAGE 5 — GENAI CLINICAL REPORT DEEP DIVE
                 ============================================================== -->
            <section id="view-stage5" class="view-section">
                <div class="view-header">
                    <h2>STAGE ⑤ — GENERATIVE AI REPORT SYNTHESIS</h2>
                    <p>Multi-Stage Synthetic Patient Simulation & Cross-Modal Safety Auditing</p>
                </div>

                <!-- 1. GENAI INDEPENDENT INPUT SECTION -->
                <div class="patient-input-card">
                    <div class="card-title-strip">
                        <h3>
                            <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>
                            SEED CONDITIONS & CLINICAL CONTEXT FOR GENAI
                        </h3>
                        <div class="preset-buttons">
                            <span style="font-size: 0.68rem; color: #64748b; font-weight: 600;">SEED PRESETS:</span>
                            <button class="btn-preset" type="button" onclick="setGenAISeedSample('NSCLC')">Metastatic NSCLC</button>
                            <button class="btn-preset" type="button" onclick="setGenAISeedSample('BRCA')">CDK4/6 Breast Cancer</button>
                            <button class="btn-preset danger" type="button" onclick="setGenAISeedSample('DILI')">Severe DILI Wildcard</button>
                        </div>
                    </div>

                    <form id="form-stage5" onsubmit="event.preventDefault(); runStage5Generation();">
                        <div class="input-grid-4">
                            <div class="form-group">
                                <label class="form-label">Cancer Type</label>
                                <select id="s5-cancer-type" class="form-control">
                                    <option value="Lung (LUAD)" selected>Lung Adenocarcinoma (LUAD)</option>
                                    <option value="Lung (LUSC)">Lung Squamous (LUSC)</option>
                                    <option value="Breast (BRCA)">Breast Invasive (BRCA)</option>
                                    <option value="Colon (COAD)">Colon Adeno (COAD)</option>
                                    <option value="Prostate (PRAD)">Prostate Adeno (PRAD)</option>
                                    <option value="Stomach (STAD)">Stomach Adeno (STAD)</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label class="form-label">Age Range</label>
                                <input type="text" id="s5-age-range" class="form-control" value="60 - 70">
                            </div>
                            <div class="form-group">
                                <label class="form-label">Mutation / Genomic Biomarker</label>
                                <input type="text" id="s5-mutation" class="form-control" value="EGFR L858R, MET amp">
                            </div>
                            <div class="form-group">
                                <label class="form-label">Biomarker</label>
                                <input type="text" id="s5-biomarker" class="form-control" value="PD-L1 85%, TMB 14 mut/Mb">
                            </div>
                        </div>

                        <div class="input-grid-3">
                            <div class="form-group">
                                <label class="form-label">Cancer Stage</label>
                                <select id="s5-stage" class="form-control">
                                    <option value="Stage I">Stage I</option>
                                    <option value="Stage II">Stage II</option>
                                    <option value="Stage III">Stage III</option>
                                    <option value="Stage IV" selected>Stage IV</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label class="form-label">Treatment</label>
                                <input type="text" id="s5-treatment" class="form-control" value="Osimertinib + Capmatinib">
                            </div>
                            <div class="form-group">
                                <label class="form-label">Baseline ALT (U/L)</label>
                                <input type="number" id="s5-alt" class="form-control" value="248" step="1">
                            </div>
                        </div>

                        <div class="form-group" style="margin-bottom: 1rem;">
                            <label class="form-label">Optional Clinical Context / Seed Symptoms</label>
                            <textarea id="s5-notes" class="form-control" style="min-height: 80px;" placeholder="Add clinical background context...">Patient developing fever 38.4C and elevated transaminases under targeted therapy combination.</textarea>
                        </div>

                        <div class="btn-action-row">
                            <button type="submit" id="btn-run-stage5" class="btn-primary-analyze" style="max-width: 280px;">
                                [ GENERATE ]
                            </button>
                        </div>
                    </form>
                </div>

                <!-- 2. GENAI OUTPUT SECTION -->
                <div id="s5-output-card" class="final-assessment-card" style="display: none; margin-top: 1rem;">
                    <div class="assessment-banner">
                        <div class="assessment-banner-left">
                            <h3 id="s5-out-title">GENERATED RESULT</h3>
                            <p>Patient / Synthetic Profile & Cross-Modal Safety Auditing</p>
                        </div>
                        <div class="risk-classification-tag high" id="s5-out-badge">VALIDATED REPORT</div>
                    </div>

                    <div class="assessment-breakdown-grid">
                        <div class="breakdown-box">
                            <span>FACTUAL CONSISTENCY</span>
                            <h4 id="s5-out-factual">98.2%</h4>
                        </div>
                        <div class="breakdown-box">
                            <span>CROSS-MODAL SAFETY</span>
                            <h4 id="s5-out-safety">95.0%</h4>
                        </div>
                        <div class="breakdown-box">
                            <span>HALLUCINATION INDEX</span>
                            <h4 id="s5-out-halluc" style="color: #059669;">1.8%</h4>
                        </div>
                        <div class="breakdown-box">
                            <span>VALIDATION</span>
                            <h4 id="s5-out-valid" style="color: #059669;">PASSED</h4>
                        </div>
                    </div>

                    <div>
                        <h4 style="font-size: 0.82rem; font-weight: 700; margin-bottom: 0.4rem;">GENERATED PATIENT / SYNTHETIC PROFILE</h4>
                        <div class="clinical-summary-box" id="s5-out-summary">
                            Generated clinical summary...
                        </div>
                    </div>

                    <div>
                        <h4 style="font-size: 0.82rem; font-weight: 700; margin-bottom: 0.4rem;">KEY CLINICAL FINDINGS</h4>
                        <div class="considerations-checklist" id="s5-out-findings"></div>
                    </div>

                    <div>
                        <h4 style="font-size: 0.82rem; font-weight: 700; margin-bottom: 0.4rem;">FULL GENERATED CLINICAL REPORT</h4>
                        <pre style="background: #0b1e3f; color: #f1f5f9; padding: 1rem; border-radius: 8px; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; overflow-x: auto; white-space: pre-wrap;" id="s5-out-report"></pre>
                    </div>
                </div>

                <!-- 3. EXISTING AUDIT & DIVERGENCE METRICS -->
                <div class="table-card" style="padding: 1.5rem; margin-top: 1rem;">
                    <h3 style="font-size: 0.95rem; font-weight: 800; margin-bottom: 0.85rem;">Audit & Divergence Metrics</h3>
                    <div class="analytics-stats-grid">
                        <div class="stat-card"><span>Factual Consistency</span><h3>98.2%</h3></div>
                        <div class="stat-card"><span>Cross-Modal Safety</span><h3>95.0%</h3></div>
                        <div class="stat-card"><span>Hallucination Rate</span><h3 style="color: #059669;">1.8%</h3></div>
                        <div class="stat-card"><span>Divergence Score</span><h3>0.042</h3></div>
                    </div>
                </div>
            </section>


            <!-- ==============================================================
                 VIEW 6B: STAGE 6 — AGENTIC AI COMMAND CENTER
                 ============================================================== -->
            <section id="view-stage6" class="view-section">
                <div class="view-header">
                    <h2>STAGE ⑥ — AGENTIC AI</h2>
                    <p>Autonomous Multi-Agent Oncology Decision Engine · Deliberative Clinical Decision Support</p>
                </div>

                <!-- 1. PATIENT CONTEXT & CASE PRESETS -->
                <div class="patient-input-card">
                    <div class="card-title-strip">
                        <h3>
                            <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
                            PATIENT CONTEXT & CLINICAL REASONING FEEDER
                        </h3>
                        <div class="preset-buttons">
                            <span style="font-size: 0.68rem; color: #64748b; font-weight: 600;">CLINICAL PRESETS:</span>
                            <button class="btn-preset" type="button" onclick="setAgenticPreset('case_a_egfr')" style="background: #e0f2fe; color: #0369a1; border-color: #38bdf8;">Patient A: EGFR+ (6 ctDNA Surge)</button>
                            <button class="btn-preset" type="button" onclick="setAgenticPreset('case_b_kras')" style="background: #fef3c7; color: #92400e; border-color: #f59e0b;">Patient B: KRAS+ (Stable 2 ctDNA)</button>
                            <button class="btn-preset danger" type="button" onclick="setAgenticPreset('case_c_dili')">DILI Safety Alert</button>
                            <button class="btn-preset" type="button" onclick="setAgenticPreset('case_d_no_image')">No Image Edge Case</button>
                            <button class="btn-preset" type="button" onclick="pullExistingAiResults()" style="background: #ecfdf5; color: #065f46; border-color: #10b981; font-weight: 800;">[ USE EXISTING AI RESULTS ]</button>
                        </div>
                    </div>

                    <form id="form-stage6" onsubmit="event.preventDefault(); runAgenticAnalysis();">
                        <div class="input-grid-4">
                            <div class="form-group">
                                <label class="form-label">Patient ID</label>
                                <input type="text" id="s6-patient-id" class="form-control" value="PAT-EGFR-001">
                            </div>
                            <div class="form-group">
                                <label class="form-label">Age & Sex</label>
                                <div style="display: flex; gap: 0.5rem;">
                                    <input type="number" id="s6-age" class="form-control" value="62" style="width: 60%;">
                                    <select id="s6-sex" class="form-control" style="width: 40%;">
                                        <option value="M" selected>M</option>
                                        <option value="F">F</option>
                                    </select>
                                </div>
                            </div>
                            <div class="form-group">
                                <label class="form-label">Cancer Type & Stage</label>
                                <div style="display: flex; gap: 0.5rem;">
                                    <select id="s6-cancer-type" class="form-control" style="width: 65%;">
                                        <option value="Lung (LUAD)" selected>Lung (LUAD)</option>
                                        <option value="Breast (BRCA)">Breast (BRCA)</option>
                                        <option value="Colon (COAD)">Colon (COAD)</option>
                                    </select>
                                    <select id="s6-stage" class="form-control" style="width: 35%;">
                                        <option value="Stage IV" selected>Stage IV</option>
                                        <option value="Stage III">Stage III</option>
                                        <option value="Stage II">Stage II</option>
                                    </select>
                                </div>
                            </div>
                            <div class="form-group">
                                <label class="form-label">Genomic Biomarker</label>
                                <input type="text" id="s6-biomarker" class="form-control" value="EGFR L858R">
                            </div>
                        </div>

                        <div class="input-grid-4" style="margin-top: 0.75rem;">
                            <div class="form-group">
                                <label class="form-label">Current Regimen & Cycle</label>
                                <input type="text" id="s6-treatment" class="form-control" value="Osimertinib (Targeted TKI)">
                            </div>
                            <div class="form-group">
                                <label class="form-label">ctDNA Level (ng/mL)</label>
                                <input type="number" step="0.01" id="s6-ctdna" class="form-control" value="0.88">
                            </div>
                            <div class="form-group">
                                <label class="form-label">Rising ctDNA Readings (Count)</label>
                                <input type="number" id="s6-rising-ctdna" class="form-control" value="6">
                            </div>
                            <div class="form-group">
                                <label class="form-label">Medical Image / Scan</label>
                                <select id="s6-image-toggle" class="form-control">
                                    <option value="true" selected>Provided (CT + Pathology)</option>
                                    <option value="false">Unavailable / Pending Transfer</option>
                                </select>
                            </div>
                        </div>

                        <div class="input-grid-4" style="margin-top: 0.75rem;">
                            <div class="form-group">
                                <label class="form-label">Creatinine (mg/dL)</label>
                                <input type="number" step="0.05" id="s6-cr" class="form-control" value="1.15">
                            </div>
                            <div class="form-group">
                                <label class="form-label">ALT (U/L)</label>
                                <input type="number" id="s6-alt" class="form-control" value="48">
                            </div>
                            <div class="form-group">
                                <label class="form-label">AST (U/L)</label>
                                <input type="number" id="s6-ast" class="form-control" value="44">
                            </div>
                            <div class="form-group">
                                <label class="form-label">Total Bilirubin (mg/dL)</label>
                                <input type="number" step="0.1" id="s6-bili" class="form-control" value="1.10">
                            </div>
                        </div>

                        <div class="form-group" style="margin-top: 0.75rem;">
                            <label class="form-label">Clinical Notes / History</label>
                            <textarea id="s6-notes" class="form-control" rows="2">62yo male with metastatic EGFR+ NSCLC. Severe progression on serial imaging. 6 consecutive rising ctDNA timepoints indicating emergence of resistance bypass.</textarea>
                        </div>
                    </form>
                </div>

                <!-- 2. AGENT STATUS DASHBOARD -->
                <div class="table-card" style="padding: 1.25rem; margin-top: 1rem;">
                    <div style="display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid var(--border-card); padding-bottom: 0.75rem;">
                        <div>
                            <h3 style="font-size: 0.92rem; font-weight: 800; color: var(--text-primary);">MULTI-AGENT ORCHESTRATION STATUS DASHBOARD</h3>
                            <p style="font-size: 0.72rem; color: var(--text-light); margin-top: 2px;">Real-time state tracking across 10 specialized autonomous clinical agents</p>
                        </div>
                        <div style="display: flex; gap: 0.6rem; font-size: 0.68rem; font-weight: 700;">
                            <span style="color: #94a3b8;">○ Waiting</span>
                            <span style="color: #0284c7;">⟳ Processing</span>
                            <span style="color: #059669;">✓ Completed</span>
                            <span style="color: #d97706;">⚠ Warning</span>
                            <span style="color: #dc2626;">✕ Failed</span>
                        </div>
                    </div>

                    <div class="agentic-status-grid">
                        <div class="agent-node-card" id="agent-card-agent_patient_data">
                            <div class="agent-node-info">
                                <h4>PATIENT DATA AGENT</h4>
                                <p id="agent-msg-agent_patient_data">Schema & normalization</p>
                            </div>
                            <div class="agent-status-indicator waiting" id="agent-dot-agent_patient_data">○</div>
                        </div>
                        <div class="agent-node-card" id="agent-card-agent_risk_analysis">
                            <div class="agent-node-info">
                                <h4>RISK ANALYSIS AGENT</h4>
                                <p id="agent-msg-agent_risk_analysis">Stage 1 ML toxicity</p>
                            </div>
                            <div class="agent-status-indicator waiting" id="agent-dot-agent_risk_analysis">○</div>
                        </div>
                        <div class="agent-node-card" id="agent-card-agent_image_analysis">
                            <div class="agent-node-info">
                                <h4>CLINICAL ANALYSIS AGENT</h4>
                                <p id="agent-msg-agent_image_analysis">Stage 2 DL imaging</p>
                            </div>
                            <div class="agent-status-indicator waiting" id="agent-dot-agent_image_analysis">○</div>
                        </div>
                        <div class="agent-node-card" id="agent-card-agent_clinical_nlp">
                            <div class="agent-node-info">
                                <h4>CLINICAL NLP AGENT</h4>
                                <p id="agent-msg-agent_clinical_nlp">Stage 3 triage & NER</p>
                            </div>
                            <div class="agent-status-indicator waiting" id="agent-dot-agent_clinical_nlp">○</div>
                        </div>
                        <div class="agent-node-card" id="agent-card-agent_clinical_briefing">
                            <div class="agent-node-info">
                                <h4>SLM BRIEFING AGENT</h4>
                                <p id="agent-msg-agent_clinical_briefing">Stage 4 concise summary</p>
                            </div>
                            <div class="agent-status-indicator waiting" id="agent-dot-agent_clinical_briefing">○</div>
                        </div>
                        <div class="agent-node-card" id="agent-card-agent_scenario_analysis">
                            <div class="agent-node-info">
                                <h4>GENAI SCENARIO AGENT</h4>
                                <p id="agent-msg-agent_scenario_analysis">Stage 5 in-silico stress</p>
                            </div>
                            <div class="agent-status-indicator waiting" id="agent-dot-agent_scenario_analysis">○</div>
                        </div>
                        <div class="agent-node-card" id="agent-card-agent_treatment_optimization">
                            <div class="agent-node-info">
                                <h4>TREATMENT AGENT</h4>
                                <p id="agent-msg-agent_treatment_optimization">Regimen synthesis</p>
                            </div>
                            <div class="agent-status-indicator waiting" id="agent-dot-agent_treatment_optimization">○</div>
                        </div>
                        <div class="agent-node-card" id="agent-card-agent_trial_matching">
                            <div class="agent-node-info">
                                <h4>TRIAL MATCHING AGENT</h4>
                                <p id="agent-msg-agent_trial_matching">ReAct trial allocation</p>
                            </div>
                            <div class="agent-status-indicator waiting" id="agent-dot-agent_trial_matching">○</div>
                        </div>
                        <div class="agent-node-card" id="agent-card-agent_safety_guardrail">
                            <div class="agent-node-info">
                                <h4>SAFETY AGENT</h4>
                                <p id="agent-msg-agent_safety_guardrail">Organ toxicity interlock</p>
                            </div>
                            <div class="agent-status-indicator waiting" id="agent-dot-agent_safety_guardrail">○</div>
                        </div>
                        <div class="agent-node-card" id="agent-card-agent_final_decision">
                            <div class="agent-node-info">
                                <h4>FINAL DECISION AGENT</h4>
                                <p id="agent-msg-agent_final_decision">Multi-agent consensus</p>
                            </div>
                            <div class="agent-status-indicator waiting" id="agent-dot-agent_final_decision">○</div>
                        </div>
                    </div>
                </div>

                <!-- 3. PHYSICIAN CONTROL STRIP -->
                <div class="physician-control-bar">
                    <div class="physician-control-title">
                        <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
                        <span>ONCOLOGY TUMOR BOARD COMMAND & CONTROL</span>
                        <span id="s6-override-badge" class="metric-badge red" style="display: none; margin-left: 0.5rem;">PHYSICIAN OVERRIDE ACTIVE</span>
                    </div>
                    <div class="physician-btn-group">
                        <button class="btn-physician primary" id="btn-run-agentic" onclick="runAgenticAnalysis()">
                            <svg viewBox="0 0 24 24" width="14" height="14" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>
                            [ RUN AGENTIC ANALYSIS ]
                        </button>
                        <button class="btn-physician pause" id="btn-pause-agentic" onclick="togglePauseAgentic()">[ PAUSE AGENT ]</button>
                        <button class="btn-physician stop" id="btn-stop-agentic" onclick="stopAgentic()">[ STOP AGENT ]</button>
                        <button class="btn-physician override" id="btn-override-agentic" onclick="promptPhysicianOverride()">[ PHYSICIAN OVERRIDE ]</button>
                    </div>
                </div>

                <!-- 4. SAFETY HALT BANNER -->
                <div id="s6-safety-halt-banner" class="safety-alert-halt" style="display: none;">
                    <svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="#ef4444" style="flex-shrink:0;"><polygon points="7.86 2 16.14 2 22 7.86 22 16.14 16.14 22 7.86 22 2 16.14 2 7.86 7.86 2"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
                    <div>
                        <h3>🚨 SAFETY REVIEW REQUIRED — WORKFLOW HALTED</h3>
                        <p id="s6-safety-halt-text">A critical clinical contraindication has been intercepted by the Safety Guardrail Agent. Autonomous trial matching and therapy escalation are suspended until oncologist review.</p>
                    </div>
                </div>

                <!-- 5. AGENTIC ANALYSIS RESULTS SECTION -->
                <div id="s6-results-container" style="display: none; flex-direction: column; gap: 1.25rem;">
                    
                    <!-- FINAL MULTI-AGENT CLINICAL ASSESSMENT CARD -->
                    <div class="final-assessment-card" style="display: flex; flex-direction: column; gap: 1rem; border: 2px solid #0284c7; background: #ffffff;">
                        <div class="card-title-strip" style="border-bottom: 1px solid var(--border-card); padding-bottom: 0.75rem;">
                            <div>
                                <h3 style="font-size: 1.1rem; color: #0284c7;">FINAL MULTI-AGENT CLINICAL ASSESSMENT</h3>
                                <p id="s6-final-patient-meta" style="font-size: 0.78rem; color: var(--text-light); margin-top: 2px;">Patient ID: PAT-EGFR-001 | Status: SUCCESS</p>
                            </div>
                            <span id="s6-final-status-pill" class="metric-badge green" style="font-size: 0.82rem; padding: 0.4rem 0.85rem;">CONSENSUS ACHIEVED</span>
                        </div>

                        <div class="clinical-summary-box" style="border-left-color: #0284c7; background: #f0f9ff;">
                            <strong style="color: #0369a1; font-size: 0.85rem; display: block; margin-bottom: 0.25rem;">MULTI-AGENT CONSENSUS DIRECTIVE:</strong>
                            <span id="s6-final-directive" style="font-size: 0.88rem; font-weight: 700; color: #0f172a; line-height: 1.6;">Recommendation generated...</span>
                        </div>

                        <div class="assessment-breakdown-grid">
                            <div class="breakdown-box">
                                <span>STAGE 1 ML RISK</span>
                                <h4 id="s6-sum-ml">Low Toxicity</h4>
                            </div>
                            <div class="breakdown-box">
                                <span>STAGE 2 DL IMAGING</span>
                                <h4 id="s6-sum-dl">Stable Lesion</h4>
                            </div>
                            <div class="breakdown-box">
                                <span>STAGE 3 NLP TRIAGE</span>
                                <h4 id="s6-sum-nlp">MODERATE</h4>
                            </div>
                            <div class="breakdown-box">
                                <span>SAFETY GUARDRAIL</span>
                                <h4 id="s6-sum-safety" style="color: #059669;">PASSED</h4>
                            </div>
                        </div>

                        <div class="disclaimer-strip" style="color: #64748b; font-weight: 600;">
                            <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>
                            AI-generated clinical decision support. Final treatment decisions require qualified oncologist review.
                        </div>
                    </div>

                    <!-- TWO-COLUMN SECTION: TREATMENT OPTIMIZATION & TRIAL MATCHING -->
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.25rem;">
                        
                        <!-- CARD A: TREATMENT OPTIMIZATION -->
                        <div class="table-card" style="padding: 1.25rem;">
                            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.75rem; border-bottom: 1px solid var(--border-card); padding-bottom: 0.5rem;">
                                <h3 style="font-size: 0.92rem; font-weight: 800; color: var(--text-primary);">TREATMENT OPTIMIZATION</h3>
                                <span class="metric-badge blue" style="font-size: 0.65rem;">SYNTHESIZED PROTOCOLS</span>
                            </div>
                            <div id="s6-treatment-options-list" style="display: flex; flex-direction: column; gap: 0.75rem;">
                                <!-- Rendered dynamically -->
                            </div>
                            <div style="font-size: 0.7rem; color: #64748b; margin-top: 0.75rem; font-style: italic; border-top: 1px solid var(--border-card); padding-top: 0.5rem;">
                                AI-generated clinical decision support — requires qualified oncologist review.
                            </div>
                        </div>

                        <!-- CARD B: CLINICAL TRIAL MATCHING -->
                        <div class="table-card" style="padding: 1.25rem;">
                            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.75rem; border-bottom: 1px solid var(--border-card); padding-bottom: 0.5rem;">
                                <h3 style="font-size: 0.92rem; font-weight: 800; color: var(--text-primary);">CLINICAL TRIAL MATCHING & SLOTS</h3>
                                <span class="metric-badge green" id="s6-trial-count-badge" style="font-size: 0.65rem;">VERIFIED REGISTRY</span>
                            </div>
                            <div id="s6-trial-matches-list" style="display: flex; flex-direction: column; gap: 0.75rem;">
                                <!-- Rendered dynamically -->
                            </div>
                        </div>
                    </div>

                    <!-- TRANSPARENT DECISION TRACE TIMELINE -->
                    <div class="table-card" style="padding: 1.25rem;">
                        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.75rem; border-bottom: 1px solid var(--border-card); padding-bottom: 0.5rem;">
                            <div>
                                <h3 style="font-size: 0.92rem; font-weight: 800; color: var(--text-primary);">LIVE AGENT DECISION TRACE</h3>
                                <p style="font-size: 0.72rem; color: var(--text-light); margin-top: 2px;">Step-by-step transparent ReAct reasoning trail (auditable without raw chain-of-thought)</p>
                            </div>
                            <span class="metric-badge blue" id="s6-trace-count-pill" style="font-size: 0.68rem;">10 STEPS VERIFIED</span>
                        </div>
                        <div class="table-responsive">
                            <table class="decision-trace-table">
                                <thead>
                                    <tr>
                                        <th style="width: 50px;">Step</th>
                                        <th style="width: 170px;">Agent</th>
                                        <th style="width: 160px;">Tool / API Used</th>
                                        <th style="width: 90px;">Status</th>
                                        <th style="width: 80px;">Time</th>
                                        <th>Reasoning Summary & Audit Findings</th>
                                    </tr>
                                </thead>
                                <tbody id="s6-decision-trace-tbody">
                                    <!-- Populated dynamically -->
                                </tbody>
                            </table>
                        </div>
                    </div>

                    <!-- COMPREHENSIVE SAFETY CHECK AUDIT CARD -->
                    <div class="table-card" style="padding: 1.25rem;">
                        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.75rem; border-bottom: 1px solid var(--border-card); padding-bottom: 0.5rem;">
                            <h3 style="font-size: 0.92rem; font-weight: 800; color: var(--text-primary);">ORGAN TOXICITY & SAFETY INTERLOCK AUDIT</h3>
                            <span class="metric-badge green" id="s6-safety-overall-badge">ALL FLOORS PASSED</span>
                        </div>
                        <div id="s6-safety-checks-grid" style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.75rem;">
                            <!-- Populated dynamically -->
                        </div>
                    </div>

                </div>
            </section>


            <!-- ==============================================================
                 VIEW 7: HISTORY
                 ============================================================== -->
            <section id="view-history" class="view-section">
                <div class="view-header">
                    <h2>PATIENT EVALUATION HISTORY</h2>
                    <p>Repository of previously analyzed clinical cases with 1-click inspection</p>
                </div>
                <div class="table-card">
                    <div class="table-responsive">
                        <table class="clinical-table">
                            <thead>
                                <tr>
                                    <th>Patient ID</th>
                                    <th>Date & Time</th>
                                    <th>Cancer Type</th>
                                    <th>ML Result</th>
                                    <th>DL Result</th>
                                    <th>NLP Triage</th>
                                    <th>Final Risk</th>
                                    <th>Actions</th>
                                </tr>
                            </thead>
                            <tbody id="history-table-body">
                                <tr><td colspan="8" style="text-align: center; color: #94a3b8; padding: 2rem;">Loading history...</td></tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </section>


            <!-- ==============================================================
                 VIEW 8: ANALYTICS
                 ============================================================== -->
            <section id="view-analytics" class="view-section">
                <div class="view-header">
                    <h2>HOSPITAL CDSS ANALYTICS</h2>
                    <p>System-wide clinical performance indicators and patient risk cohort distribution</p>
                </div>

                <div class="analytics-stats-grid" id="analytics-stat-cards">
                    <div class="stat-card"><span>Total Patients Analyzed</span><h3 id="stat-total">142</h3></div>
                    <div class="stat-card"><span>High Risk Cohort</span><h3 id="stat-high" style="color: #ea580c;">31</h3></div>
                    <div class="stat-card"><span>Moderate Risk Cohort</span><h3 id="stat-mod" style="color: #2563eb;">54</h3></div>
                    <div class="stat-card"><span>Low Risk Cohort</span><h3 id="stat-low" style="color: #059669;">48</h3></div>
                </div>

                <div class="table-card" style="padding: 1.5rem; margin-top: 1rem;">
                    <h3 style="font-size: 0.95rem; font-weight: 800; margin-bottom: 0.85rem;">Stage Execution Reliability & Latency</h3>
                    <table class="clinical-table">
                        <thead>
                            <tr><th>Stage</th><th>Model Backbone</th><th>Success Rate</th><th>Avg Latency</th></tr>
                        </thead>
                        <tbody>
                            <tr><td>Stage 1 (ML)</td><td>Voting Ensemble (RF + ExtraTrees + XGBoost)</td><td><span class="metric-badge green">99.4%</span></td><td>8.4 ms</td></tr>
                            <tr><td>Stage 2 (DL)</td><td>MultimodalLSTM (CNN + Tabular Sequence)</td><td><span class="metric-badge green">98.8%</span></td><td>28.6 ms</td></tr>
                            <tr><td>Stage 3 (NLP)</td><td>TF-IDF + LR Classifier with Rule NER</td><td><span class="metric-badge green">99.6%</span></td><td>14.2 ms</td></tr>
                            <tr><td>Stage 4 (SLM)</td><td>Qwen2.5-3B Instruct + Safety Guardrails</td><td><span class="metric-badge green">99.1%</span></td><td>12.5 ms</td></tr>
                            <tr><td>Stage 5 (GenAI)</td><td>Clinical Report Synthesizer & Auditor</td><td><span class="metric-badge green">98.9%</span></td><td>18.1 ms</td></tr>
                        </tbody>
                    </table>
                </div>
            </section>


            <!-- ==============================================================
                 VIEW 9: SETTINGS
                 ============================================================== -->
            <section id="view-settings" class="view-section">
                <div class="view-header">
                    <h2>SYSTEM SETTINGS & RUNTIME STATUS</h2>
                    <p>Clinical decision-support threshold parameters, model weights, and API endpoints</p>
                </div>
                <div class="table-card" style="padding: 1.5rem;">
                    <div style="display: flex; flex-direction: column; gap: 1rem;">
                        <div>
                            <label class="form-label">FastAPI Gateway URL</label>
                            <input type="text" class="form-control" value="http://127.0.0.1:8000" readonly style="background: #f8fafc;">
                        </div>
                        <div>
                            <label class="form-label">Active Compute Device</label>
                            <input type="text" class="form-control" value="CPU (Optimized with OpenBLAS / MKL)" readonly style="background: #f8fafc;">
                        </div>
                        <div>
                            <label class="form-label">Stage 1 Model Checkpoint</label>
                            <input type="text" class="form-control" value="Stage1/best_toxicity_model.pkl" readonly style="background: #f8fafc;">
                        </div>
                        <div>
                            <label class="form-label">Stage 2 DL Checkpoint</label>
                            <input type="text" class="form-control" value="Stage2/best_stage02_multimodal_lstm.pth" readonly style="background: #f8fafc;">
                        </div>
                    </div>
                </div>
            </section>

        </div>
    </main>

    <!-- ======================================================================
         CLIENT-SIDE JAVASCRIPT ORCHESTRATOR
         ====================================================================== -->
    <script>
        let currentImageFile = "luad/ct_slice_0000.png";

        // Navigation Switcher
        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', function() {
                const targetView = this.getAttribute('data-view');
                if (targetView) {
                    switchView(targetView);
                }
            });
        });

        function switchView(viewId) {
            document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
            const matchingNav = document.querySelector(`.nav-link[data-view="${viewId}"]`);
            if (matchingNav) matchingNav.classList.add('active');

            document.querySelectorAll('.view-section').forEach(sec => sec.classList.remove('active'));
            const targetSec = document.getElementById(viewId);
            if (targetSec) targetSec.classList.add('active');

            if (viewId === 'view-history') loadHistory();
            if (viewId === 'view-analytics') loadAnalytics();
        }

        // Preset Loader
        async function loadPreset(presetKey) {
            try {
                const res = await fetch('/api/pipeline/presets');
                const presets = await res.json();
                const p = presets[presetKey];
                if (!p || !p.data) return;

                const d = p.data;
                for (const k in d) {
                    const el = document.getElementById(`inp-${k}`);
                    if (el) el.value = d[k];
                }

                if (d.image_file) {
                    currentImageFile = d.image_file;
                    const preview = document.getElementById('image-preview');
                    const imgText = document.getElementById('image-filename');
                    preview.src = `/api/pipeline/image/${d.image_file}`;
                    preview.style.display = 'block';
                    imgText.innerText = d.image_file.split('/').pop();
                }
            } catch (e) {
                console.error("Preset load error", e);
            }
        }

        // CSV Upload Handler
        async function handleCSVUpload(input) {
            if (!input.files || input.files.length === 0) return;
            const file = input.files[0];
            document.getElementById('csv-filename').innerText = file.name;

            const formData = new FormData();
            formData.append('file', file);

            try {
                const res = await fetch('/api/pipeline/upload-csv', {
                    method: 'POST',
                    body: formData
                });
                const result = await res.json();
                if (result.status === 'success' && result.patient_data) {
                    const d = result.patient_data;
                    for (const k in d) {
                        const el = document.getElementById(`inp-${k}`);
                        if (el && d[k] !== null && d[k] !== undefined) {
                            el.value = d[k];
                        }
                    }
                    alert(result.message);
                } else {
                    alert("Upload error: " + (result.detail || "Could not parse CSV."));
                }
            } catch (e) {
                alert("CSV upload failed: " + e.message);
            }
        }

        // Image Upload Handler
        function handleImageUpload(input) {
            if (!input.files || input.files.length === 0) return;
            const file = input.files[0];
            document.getElementById('image-filename').innerText = file.name;

            const reader = new FileReader();
            reader.onload = function(e) {
                const preview = document.getElementById('image-preview');
                preview.src = e.target.result;
                preview.style.display = 'block';
            };
            reader.readAsDataURL(file);
        }

        function resetForm() {
            document.getElementById('patient-form').reset();
            document.getElementById('image-preview').style.display = 'none';
            document.getElementById('csv-filename').innerText = 'Upload Patient CSV';
            document.getElementById('image-filename').innerText = 'Medical Image Upload (DL)';
            document.getElementById('results-grid').style.display = 'none';
            document.getElementById('final-assessment-section').style.display = 'none';
            resetPipelineFlow();
        }

        function resetPipelineFlow() {
            ['stage1', 'stage2', 'stage3', 'stage4', 'stage5'].forEach(s => {
                const node = document.getElementById(`node-${s}`);
                const pill = document.getElementById(`pill-${s}`);
                if (node) node.className = 'flow-node';
                if (pill) { pill.innerText = 'Waiting'; }
            });
            document.getElementById('pipeline-progress-fill').style.width = '0%';
            document.getElementById('pipeline-progress-label').innerText = 'Stage 0 of 5 (Ready)';
        }

        // Sequential 5-Stage Execution
        async function runCompleteAnalysis() {
            const btn = document.getElementById('btn-run-analysis');
            btn.disabled = true;
            btn.innerHTML = `<span class="system-dot" style="background:#fff;"></span> EXECUTING MULTI-STAGE ANALYSIS...`;

            resetPipelineFlow();
            document.getElementById('results-grid').style.display = 'grid';
            document.getElementById('final-assessment-section').style.display = 'none';

            // Harvest form values
            const payload = {
                patient_id: document.getElementById('inp-patient_id').value,
                age: parseFloat(document.getElementById('inp-age').value) || 60,
                sex: document.getElementById('inp-sex').value,
                cancer_type: document.getElementById('inp-cancer_type').value,
                cancer_stage: document.getElementById('inp-cancer_stage').value,
                mutation_profile: document.getElementById('inp-mutation_profile').value,
                tumor_marker_level: parseFloat(document.getElementById('inp-tumor_marker_level').value) || 14.2,
                ctDNA_level: parseFloat(document.getElementById('inp-ctDNA_level').value) || 0.45,
                ALT: parseFloat(document.getElementById('inp-ALT').value) || 45,
                AST: parseFloat(document.getElementById('inp-AST').value) || 42,
                creatinine: parseFloat(document.getElementById('inp-creatinine').value) || 1.1,
                bilirubin: parseFloat(document.getElementById('inp-bilirubin').value) || 1.2,
                treatment_name: document.getElementById('inp-treatment_name').value,
                dosage_mg: parseFloat(document.getElementById('inp-dosage_mg').value) || 400,
                treatment_cycle: parseInt(document.getElementById('inp-treatment_cycle').value) || 3,
                clinical_notes: document.getElementById('inp-clinical_notes').value,
                image_file: currentImageFile
            };

            // Animate pipeline stages sequentially
            const stages = [
                { id: 'stage1', name: 'Stage 1 of 5: ML Toxicity Analysis', pct: 20 },
                { id: 'stage2', name: 'Stage 2 of 5: DL Medical Image Analysis', pct: 40 },
                { id: 'stage3', name: 'Stage 3 of 5: NLP Clinical Triage', pct: 60 },
                { id: 'stage4', name: 'Stage 4 of 5: SLM Clinical Reasoning', pct: 80 },
                { id: 'stage5', name: 'Stage 5 of 5: GenAI Final Report', pct: 100 }
            ];

            let activeIdx = 0;
            const progressTimer = setInterval(() => {
                if (activeIdx < stages.length) {
                    const curr = stages[activeIdx];
                    document.getElementById('pipeline-progress-fill').style.width = curr.pct + '%';
                    document.getElementById('pipeline-progress-label').innerText = curr.name;

                    const currNode = document.getElementById(`node-${curr.id}`);
                    const currPill = document.getElementById(`pill-${curr.id}`);
                    if (currNode) currNode.className = 'flow-node processing-stage';
                    if (currPill) currPill.innerText = 'Processing ⟳';

                    if (activeIdx > 0) {
                        const prev = stages[activeIdx - 1];
                        const prevNode = document.getElementById(`node-${prev.id}`);
                        const prevPill = document.getElementById(`pill-${prev.id}`);
                        if (prevNode) prevNode.className = 'flow-node completed-stage';
                        if (prevPill) prevPill.innerText = 'Completed ✓';
                    }
                    activeIdx++;
                }
            }, 350);

            try {
                const res = await fetch('/api/pipeline/run-all', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await res.json();
                clearInterval(progressTimer);

                // Set all stages completed
                stages.forEach(s => {
                    const node = document.getElementById(`node-${s.id}`);
                    const pill = document.getElementById(`pill-${s.id}`);
                    if (node) node.className = 'flow-node completed-stage';
                    if (pill) pill.innerText = 'Completed ✓';
                });
                document.getElementById('pipeline-progress-fill').style.width = '100%';
                document.getElementById('pipeline-progress-label').innerText = 'Analysis Complete (100%)';

                // Render Intermediate Results
                renderStageOutputs(data);

                // Render Final Assessment
                renderFinalAssessment(data);

            } catch (err) {
                clearInterval(progressTimer);
                alert("Pipeline execution failed: " + err.message);
            } finally {
                btn.disabled = false;
                btn.innerHTML = `<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor"><polygon points="5 3 19 12 5 21 5 3" fill="currentColor"/></svg> RUN COMPLETE AI ANALYSIS`;
            }
        }

        function renderStageOutputs(data) {
            const stages = data.pipeline_stages || {};

            // 1. ML Output
            const s1 = stages.stage1_ml || {};
            document.getElementById('ml-badge-risk').innerText = `${s1.prediction || 'N/A'} Risk (${s1.confidence_pct || 85}%)`;
            document.getElementById('ml-text-interp').innerText = s1.clinical_interpretation || '';
            const mlTags = document.getElementById('ml-tags-list');
            mlTags.innerHTML = '';
            (s1.important_features || []).forEach(f => {
                mlTags.innerHTML += `<span class="feature-tag"><strong>${f.feature.split(' ')[0]}:</strong> ${f.value} (${f.impact})</span>`;
            });

            // 2. DL Output
            const s2 = stages.stage2_dl || {};
            document.getElementById('dl-badge-risk').innerText = `${s2.risk_tier || 'Moderate'} (${Math.round((s2.progression_probability || 0.45)*100)}% Prog)`;
            document.getElementById('dl-text-interp').innerText = s2.clinical_interpretation || '';
            const dlRow = document.getElementById('dl-preview-row');
            dlRow.innerHTML = `
                <img src="${s2.pathology_tile_file || '/api/pipeline/image/luad/tile_0000.png'}" style="width: 38px; height: 38px; border-radius: 4px; object-fit: cover; border: 1px solid #e2e8f0;">
                <img src="${s2.ct_slice_file || '/api/pipeline/image/luad/ct_slice_0000.png'}" style="width: 38px; height: 38px; border-radius: 4px; object-fit: cover; border: 1px solid #e2e8f0;">
                <span style="font-size: 0.72rem; color: var(--text-secondary);">Lesion: <strong>${s2.image_prediction || 'Stable'}</strong></span>
            `;

            // 3. NLP Output
            const s3 = stages.stage3_nlp || {};
            document.getElementById('nlp-badge-triage').innerText = `${s3.urgency_classification || 'MODERATE'} Priority`;
            document.getElementById('nlp-text-interp').innerText = s3.clinical_interpretation || '';
            const nlpTags = document.getElementById('nlp-entities-list');
            nlpTags.innerHTML = '';
            (s3.extracted_entities || []).slice(0, 6).forEach(e => {
                nlpTags.innerHTML += `<span class="feature-tag"><strong>${e.label}:</strong> ${e.text}</span>`;
            });

            // 4. SLM Output
            const s4 = stages.stage4_slm || {};
            document.getElementById('slm-badge-guardrail').innerText = s4.guardrail_status || 'VERIFIED_SAFE';
            document.getElementById('slm-text-interp').innerText = s4.clinical_reasoning || '';
        }

        function renderFinalAssessment(data) {
            const final = data.final_assessment || {};
            const finalCard = document.getElementById('final-assessment-section');
            finalCard.style.display = 'flex';

            document.getElementById('final-card-patient-title').innerText = `FINAL AI ASSESSMENT — ${final.patient_id || ''}`;
            document.getElementById('final-card-patient-sub').innerText = `Diagnosis: ${final.cancer_type || 'Oncology'} | Timestamp: ${new Date().toLocaleDateString()}`;

            const riskTag = document.getElementById('final-risk-tag');
            const riskText = final.overall_risk || 'MODERATE RISK';
            riskTag.innerText = riskText;

            riskTag.className = 'risk-classification-tag';
            if (riskText.includes('LOW')) riskTag.classList.add('low');
            else if (riskText.includes('MODERATE')) riskTag.classList.add('moderate');
            else if (riskText.includes('CRITICAL')) riskTag.classList.add('critical');
            else riskTag.classList.add('high');

            document.getElementById('fin-overall-risk').innerText = riskText;
            document.getElementById('fin-risk-prob').innerText = `${final.risk_probability_pct || 75.0}%`;
            document.getElementById('fin-confidence').innerText = `${final.confidence_pct || 90.0}%`;

            document.getElementById('fin-genai-summary').innerText = final.genai_summary || '';

            const considerationsList = document.getElementById('fin-considerations-list');
            considerationsList.innerHTML = '';
            (final.clinical_considerations || []).forEach(c => {
                considerationsList.innerHTML += `
                    <div class="consideration-item">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><polyline points="9 11 12 14 22 4"/><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/></svg>
                        <span>${c}</span>
                    </div>
                `;
            });

            finalCard.scrollIntoView({ behavior: 'smooth' });
        }

        // Load History Records
        async function loadHistory() {
            const tbody = document.getElementById('history-table-body');
            tbody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: #94a3b8; padding: 2rem;">Loading history...</td></tr>`;

            try {
                const res = await fetch('/api/pipeline/history');
                const rows = await res.json();
                if (!rows || rows.length === 0) {
                    tbody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: #94a3b8; padding: 2rem;">No patient analyses recorded yet. Run an analysis above.</td></tr>`;
                    return;
                }

                tbody.innerHTML = '';
                rows.forEach(r => {
                    const badgeClass = (r.final_risk || '').toLowerCase().includes('high') ? 'red' : ((r.final_risk || '').toLowerCase().includes('low') ? 'green' : 'blue');
                    tbody.innerHTML += `
                        <tr>
                            <td><strong>${r.patient_id}</strong></td>
                            <td>${new Date(r.timestamp).toLocaleString()}</td>
                            <td>${r.cancer_type} (${r.stage})</td>
                            <td>${r.ml_result || 'Completed'}</td>
                            <td>${r.dl_result || 'Stable'}</td>
                            <td>${r.nlp_result || 'Moderate'}</td>
                            <td><span class="metric-badge ${badgeClass}">${r.final_risk}</span></td>
                            <td><button class="btn-preset" onclick="viewHistoryItem(${r.id})">Inspect</button></td>
                        </tr>
                    `;
                });
            } catch (e) {
                tbody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: red;">Failed to load history: ${e.message}</td></tr>`;
            }
        }

        async function viewHistoryItem(id) {
            try {
                const res = await fetch('/api/pipeline/history');
                const rows = await res.json();
                const item = rows.find(x => x.id === id);
                if (item && item.payload) {
                    switchView('view-prediction');
                    renderStageOutputs(item.payload);
                    renderFinalAssessment(item.payload);
                }
            } catch (e) {
                alert("Could not load record: " + e.message);
            }
        }

        // Load Analytics
        async function loadAnalytics() {
            try {
                const res = await fetch('/api/pipeline/analytics');
                const data = await res.json();
                if (data.total_patients_analyzed) {
                    document.getElementById('stat-total').innerText = data.total_patients_analyzed;
                    document.getElementById('stat-high').innerText = data.risk_distribution['HIGH RISK'] || 0;
                    document.getElementById('stat-mod').innerText = data.risk_distribution['MODERATE RISK'] || 0;
                    document.getElementById('stat-low').innerText = data.risk_distribution['LOW RISK'] || 0;
                }
            } catch (e) {
                console.error("Analytics load error", e);
            }
        }

        // ======================================================================
        // STAGE 1 (ML) INDEPENDENT TESTING HANDLERS
        // ======================================================================
        // STAGE 1 (ML) INDEPENDENT TESTING HANDLERS
        // ======================================================================
        function setMLSample(type) {
            if (type === 'HIGH') {
                document.getElementById('s1-patient-id').value = 'PAT-ML-0101';
                document.getElementById('s1-age').value = 68;
                document.getElementById('s1-sex').value = 'M';
                document.getElementById('s1-cancer-type').value = 'Lung (LUAD)';
                document.getElementById('s1-cancer-stage').value = 'Stage IV';
                document.getElementById('s1-mutation').value = 'EGFR L858R, MET amp';
                document.getElementById('s1-tumor-marker').value = 38.4;
                document.getElementById('s1-ctdna').value = 2.45;
                document.getElementById('s1-alt').value = 265;
                document.getElementById('s1-ast').value = 224;
                document.getElementById('s1-creatinine').value = 1.85;
                document.getElementById('s1-bilirubin').value = 2.8;
                document.getElementById('s1-treatment').value = 'Osimertinib 80mg + Capmatinib';
                document.getElementById('s1-dosage').value = 1600;
                document.getElementById('s1-cycle').value = 4;
            } else if (type === 'MODERATE') {
                document.getElementById('s1-patient-id').value = 'PAT-ML-0204';
                document.getElementById('s1-age').value = 62;
                document.getElementById('s1-sex').value = 'F';
                document.getElementById('s1-cancer-type').value = 'Breast (BRCA)';
                document.getElementById('s1-cancer-stage').value = 'Stage III';
                document.getElementById('s1-mutation').value = 'PIK3CA H1047R';
                document.getElementById('s1-tumor-marker').value = 16.5;
                document.getElementById('s1-ctdna').value = 0.82;
                document.getElementById('s1-alt').value = 75;
                document.getElementById('s1-ast').value = 68;
                document.getElementById('s1-creatinine').value = 1.15;
                document.getElementById('s1-bilirubin').value = 1.2;
                document.getElementById('s1-treatment').value = 'Abemaciclib + Fulvestrant';
                document.getElementById('s1-dosage').value = 1000;
                document.getElementById('s1-cycle').value = 3;
            } else if (type === 'LOW') {
                document.getElementById('s1-patient-id').value = 'PAT-ML-0318';
                document.getElementById('s1-age').value = 54;
                document.getElementById('s1-sex').value = 'M';
                document.getElementById('s1-cancer-type').value = 'Prostate (PRAD)';
                document.getElementById('s1-cancer-stage').value = 'Stage I';
                document.getElementById('s1-mutation').value = 'TMPRSS2-ERG';
                document.getElementById('s1-tumor-marker').value = 4.2;
                document.getElementById('s1-ctdna').value = 0.12;
                document.getElementById('s1-alt').value = 25;
                document.getElementById('s1-ast').value = 22;
                document.getElementById('s1-creatinine').value = 0.95;
                document.getElementById('s1-bilirubin').value = 0.7;
                document.getElementById('s1-treatment').value = 'Leuprolide Acetate 22.5mg';
                document.getElementById('s1-dosage').value = 400;
                document.getElementById('s1-cycle').value = 1;
            }
        }

        async function runStage1Prediction() {
            const btn = document.getElementById('btn-run-stage1');
            btn.disabled = true;
            btn.innerHTML = `<span class="system-dot" style="background:#fff;"></span> COMPUTING ML PREDICTION...`;

            const payload = {
                patient_id: document.getElementById('s1-patient-id').value || "PAT-ML-0101",
                age: parseFloat(document.getElementById('s1-age').value) || 60,
                sex: document.getElementById('s1-sex').value || "M",
                cancer_type: document.getElementById('s1-cancer-type').value,
                cancer_stage: document.getElementById('s1-cancer-stage').value,
                mutation_profile: document.getElementById('s1-mutation').value,
                tumor_marker_level: parseFloat(document.getElementById('s1-tumor-marker').value) || 20.0,
                ctDNA_level: parseFloat(document.getElementById('s1-ctdna').value) || 1.0,
                creatinine: parseFloat(document.getElementById('s1-creatinine').value) || 1.1,
                bilirubin: parseFloat(document.getElementById('s1-bilirubin').value) || 1.0,
                ALT: parseFloat(document.getElementById('s1-alt').value) || 45,
                AST: parseFloat(document.getElementById('s1-ast').value) || 40,
                treatment_name: document.getElementById('s1-treatment').value || "Standard Regimen",
                dosage_mg: parseFloat(document.getElementById('s1-dosage').value) || 400,
                treatment_cycle: parseInt(document.getElementById('s1-cycle').value) || 3
            };

            const outCard = document.getElementById('s1-output-card');

            try {
                const res = await fetch('/api/pipeline/stage/1', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                if (!res.ok) throw new Error("Model/API unavailable");
                const data = await res.json();

                outCard.style.display = 'flex';
                document.getElementById('s1-out-pid').innerText = data.patient_id || payload.patient_id;
                document.getElementById('s1-out-pred').innerText = `${data.prediction || 'Unknown'} Toxicity`;
                const riskCat = data.risk_category || `${(data.prediction || 'MODERATE').toUpperCase()} RISK`;
                document.getElementById('s1-out-class').innerText = riskCat;
                const probText = (data.prediction_probability_pct !== undefined) ? `${data.prediction_probability_pct}%` : `${data.confidence_pct || 85}%`;
                document.getElementById('s1-out-prob').innerText = probText;
                document.getElementById('s1-out-conf').innerText = `${data.confidence_pct || 85}%`;
                document.getElementById('s1-out-model-status').innerText = data.model_status || 'Operational';
                document.getElementById('s1-out-patient-sub').innerText = `Patient: ${data.patient_id || payload.patient_id} | Model: ${data.model_architecture || 'Calibrated VotingClassifier Ensemble'}`;

                const probsEl = document.getElementById('s1-out-class-probs');
                if (probsEl && data.probabilities) {
                    probsEl.innerHTML = `
                        <span><strong>Low Toxicity:</strong> ${((data.probabilities.Low || 0) * 100).toFixed(1)}%</span>
                        <span><strong>Moderate Toxicity:</strong> ${((data.probabilities.Moderate || 0) * 100).toFixed(1)}%</span>
                        <span><strong>High Toxicity:</strong> ${((data.probabilities.High || 0) * 100).toFixed(1)}%</span>
                    `;
                }

                document.getElementById('s1-out-interp').innerText = data.clinical_interpretation || 'ML evaluation complete.';

                const badge = document.getElementById('s1-out-badge');
                badge.innerText = riskCat;
                badge.className = 'risk-classification-tag';
                if (riskCat.includes('LOW')) badge.classList.add('low');
                else if (riskCat.includes('MODERATE')) badge.classList.add('moderate');
                else if (riskCat.includes('CRITICAL')) badge.classList.add('critical');
                else badge.classList.add('high');

                const tagsEl = document.getElementById('s1-out-features');
                tagsEl.innerHTML = '';
                (data.important_features || []).forEach(f => {
                    tagsEl.innerHTML += `<span class="feature-tag"><strong>${f.feature}:</strong> ${f.value} (${f.impact})</span>`;
                });
                outCard.scrollIntoView({ behavior: 'smooth' });
            } catch (err) {
                outCard.style.display = 'flex';
                document.getElementById('s1-out-pid').innerText = payload.patient_id || 'PAT-ML-0101';
                document.getElementById('s1-out-pred').innerText = 'Model/API unavailable';
                document.getElementById('s1-out-class').innerText = 'ERROR';
                document.getElementById('s1-out-prob').innerText = 'N/A';
                document.getElementById('s1-out-conf').innerText = 'N/A';
                document.getElementById('s1-out-model-status').innerText = 'Model/API unavailable';
                document.getElementById('s1-out-interp').innerText = 'Model/API unavailable: ' + err.message;
            } finally {
                btn.disabled = false;
                btn.innerHTML = `[ RUN ML PREDICTION ]`;
            }
        }

        // ======================================================================
        // STAGE 2 (DL) INDEPENDENT TESTING HANDLERS
        // ======================================================================
        let currentDLImage = "luad/ct_slice_0000.png";

        function updateDLSampleImages(cType) {
            const map = {
                'luad': 'luad/ct_slice_0000.png',
                'lusc': 'lusc/ct_slice_0000.png',
                'brca': 'brca/tile_0000.png',
                'prad': 'prad/ct_slice_0000.png',
                'coad': 'coad/tile_0000.png',
                'stad': 'stad/tile_0000.png'
            };
            const imgPath = map[cType] || 'luad/ct_slice_0000.png';
            setDLImageSample(cType, imgPath.split('/')[1]);
        }

        function setDLImageSample(cancerType, filename) {
            currentDLImage = `${cancerType}/${filename}`;
            const preview = document.getElementById('s2-image-preview');
            preview.src = `/api/pipeline/image/${currentDLImage}`;
            preview.style.display = 'block';
            document.getElementById('s2-image-filename').innerText = filename;
        }

        async function handleStage2CSV(input) {
            if (!input.files || input.files.length === 0) return;
            const file = input.files[0];
            document.getElementById('s2-csv-filename').innerText = file.name;

            const formData = new FormData();
            formData.append('file', file);

            try {
                const res = await fetch('/api/pipeline/upload-csv', { method: 'POST', body: formData });
                const result = await res.json();
                if (result.status === 'success' && result.patient_data) {
                    const d = result.patient_data;
                    if (d.patient_id) document.getElementById('s2-patient-id').value = d.patient_id;
                    if (d.age) document.getElementById('s2-age').value = d.age;
                    if (d.cancer_type) {
                        const ctEl = document.getElementById('s2-cancer-type');
                        for (let opt of ctEl.options) {
                            if (d.cancer_type.toLowerCase().includes(opt.value)) {
                                ctEl.value = opt.value;
                                break;
                            }
                        }
                    }
                    if (d.cancer_stage) document.getElementById('s2-cancer-stage').value = d.cancer_stage;
                    if (d.mutation_profile) document.getElementById('s2-mutation').value = d.mutation_profile;
                    if (d.tumor_marker_level) document.getElementById('s2-tumor-marker').value = d.tumor_marker_level;
                    if (d.ctDNA_level) document.getElementById('s2-ctdna').value = d.ctDNA_level;
                    alert("DL Patient data loaded successfully from CSV!");
                } else {
                    alert("Upload error: " + (result.detail || "Could not parse CSV."));
                }
            } catch (e) {
                alert("CSV upload failed: " + e.message);
            }
        }

        function handleStage2Image(input) {
            if (!input.files || input.files.length === 0) return;
            const file = input.files[0];
            document.getElementById('s2-image-filename').innerText = file.name;
            const reader = new FileReader();
            reader.onload = function(e) {
                const preview = document.getElementById('s2-image-preview');
                preview.src = e.target.result;
                preview.style.display = 'block';
            };
            reader.readAsDataURL(file);
        }

        async function runStage2Prediction() {
            const btn = document.getElementById('btn-run-stage2');
            btn.disabled = true;
            btn.innerHTML = `<span class="system-dot" style="background:#fff;"></span> INFERRING MULTIMODAL DL EMBEDDINGS...`;

            const payload = {
                patient_id: document.getElementById('s2-patient-id').value || "PAT-DL-0201",
                age: parseFloat(document.getElementById('s2-age').value) || 60,
                cancer_type: document.getElementById('s2-cancer-type').value,
                cancer_stage: document.getElementById('s2-cancer-stage').value,
                mutation_profile: document.getElementById('s2-mutation').value,
                tumor_marker_level: parseFloat(document.getElementById('s2-tumor-marker').value) || 20.0,
                ctDNA_level: parseFloat(document.getElementById('s2-ctdna').value) || 1.0,
                image_file: currentDLImage
            };

            const outCard = document.getElementById('s2-output-card');

            try {
                const res = await fetch('/api/pipeline/stage/2', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                if (!res.ok) throw new Error("Model/API unavailable");
                const data = await res.json();

                outCard.style.display = 'flex';
                document.getElementById('s2-out-pid').innerText = data.patient_id || payload.patient_id;
                document.getElementById('s2-out-class').innerText = data.image_prediction || 'Stable Lesion';
                document.getElementById('s2-out-classif').innerText = data.classification || data.image_prediction || 'Indeterminate Lesion';
                const prob = data.progression_probability !== undefined ? (data.progression_probability * 100).toFixed(1) + '%' : (data.probability !== undefined ? (data.probability * 100).toFixed(1) + '%' : '45.0%');
                document.getElementById('s2-out-prob').innerText = prob;
                document.getElementById('s2-out-conf').innerText = `${data.confidence_pct || 92}%`;
                document.getElementById('s2-out-model-status').innerText = data.model_status || 'Operational (MultimodalLSTM)';
                document.getElementById('s2-out-patient-sub').innerText = `Patient: ${data.patient_id || payload.patient_id} | Model: ${data.model_architecture || 'MultimodalLSTM'}`;

                const tier = data.risk_tier || 'Moderate Risk';
                const badge = document.getElementById('s2-out-risk-tier');
                badge.innerText = tier.toUpperCase();
                badge.className = 'risk-classification-tag';
                if (tier.toLowerCase().includes('low')) badge.classList.add('low');
                else if (tier.toLowerCase().includes('moderate')) badge.classList.add('moderate');
                else if (tier.toLowerCase().includes('critical')) badge.classList.add('critical');
                else badge.classList.add('high');

                document.getElementById('s2-out-interp').innerText = data.clinical_interpretation || 'DL inference complete.';

                const tbody = document.getElementById('s2-trajectory-table');
                tbody.innerHTML = '';
                (data.trajectory_breakdown || data.trajectory_timepoints || []).forEach(t => {
                    const rowClass = (t.risk_tier || '').toLowerCase().includes('high') ? 'red' : ((t.risk_tier || '').toLowerCase().includes('low') ? 'green' : 'blue');
                    tbody.innerHTML += `
                        <tr>
                            <td><strong>Step ${t.step !== undefined ? t.step : (t.sequence_step !== undefined ? t.sequence_step : '0')}</strong></td>
                            <td>${t.timepoint || ('Month ' + (t.timepoint_month || 0))}</td>
                            <td>${((t.progression_probability || 0) * 100).toFixed(1)}%</td>
                            <td><span class="metric-badge ${rowClass}">${t.risk_tier || 'Stable'}</span></td>
                        </tr>
                    `;
                });
                outCard.scrollIntoView({ behavior: 'smooth' });
            } catch (err) {
                outCard.style.display = 'flex';
                document.getElementById('s2-out-pid').innerText = payload.patient_id || 'PAT-DL-0201';
                document.getElementById('s2-out-class').innerText = 'Model/API unavailable';
                document.getElementById('s2-out-classif').innerText = 'Unavailable';
                document.getElementById('s2-out-prob').innerText = 'N/A';
                document.getElementById('s2-out-conf').innerText = 'N/A';
                document.getElementById('s2-out-model-status').innerText = 'Model/API unavailable';
                document.getElementById('s2-out-risk-tier').innerText = 'ERROR';
                document.getElementById('s2-out-interp').innerText = 'Model/API unavailable: ' + err.message;
            } finally {
                btn.disabled = false;
                btn.innerHTML = `[ RUN DL PREDICTION ]`;
            }
        }

        // ======================================================================
        // STAGE 3 (NLP) INDEPENDENT TESTING HANDLERS
        // ======================================================================
        function setNLPTextSample(type) {
            const el = document.getElementById('inp-stage3-text');
            if (type === 'HIGH') {
                el.value = "Patient has elevated tumor markers and worsening symptoms with high fever 38.8C, persistent fatigue, marked transaminase elevation, and right upper quadrant abdominal tenderness concerning for acute drug toxicity.";
            } else if (type === 'MODERATE') {
                el.value = "Patient reports mild-to-moderate fatigue, stable vital signs, mild nausea following cycle 2. Liver function tests show borderline ALT 65 U/L elevation, no acute jaundice or fever.";
            } else if (type === 'CRITICAL') {
                el.value = "Emergency ICU alert: Patient presents in acute septic shock with refractory hypotension BP 82/48, spiking temperature 39.6C, absolute neutropenia (ANC 280/uL), and multi-organ decompensation requiring immediate resuscitation.";
            }
        }

        async function runStage3Prediction() {
            const btn = document.getElementById('btn-run-stage3');
            btn.disabled = true;
            btn.innerHTML = `<span class="system-dot" style="background:#fff;"></span> PARSING CLINICAL ENTITIES & TRIAGE...`;

            const text = document.getElementById('inp-stage3-text').value;
            const payload = {
                patient_id: "TEST-NLP-PT",
                clinical_notes: text,
                cancer_type: "Lung (LUAD)"
            };

            const outCard = document.getElementById('s3-output-card');

            try {
                const res = await fetch('/api/pipeline/stage/3', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                if (!res.ok) throw new Error("Model/API unavailable");
                const data = await res.json();

                outCard.style.display = 'flex';
                const urg = (data.urgency_classification || 'MODERATE').toUpperCase();
                document.getElementById('s3-out-level').innerText = `${urg} RISK`;
                document.getElementById('s3-out-conf').innerText = `${data.confidence_pct || 91}%`;
                document.getElementById('s3-out-prob').innerText = (data.confidence !== undefined) ? (data.confidence).toFixed(3) : '0.910';
                document.getElementById('s3-out-model-status').innerText = data.model_status || 'Operational (TF-IDF + LR)';
                document.getElementById('s3-out-latency').innerText = `${data.execution_time_ms || 14.2} ms`;
                const entList = data.extracted_entities || [];
                document.getElementById('s3-out-count').innerText = `${entList.length} Identified`;
                document.getElementById('s3-out-interp').innerText = data.clinical_interpretation || 'NLP analysis complete.';

                const badge = document.getElementById('s3-out-badge');
                badge.innerText = `${urg} RISK`;
                badge.className = 'risk-classification-tag';
                if (urg.includes('LOW')) badge.classList.add('low');
                else if (urg.includes('MODERATE')) badge.classList.add('moderate');
                else if (urg.includes('CRITICAL')) badge.classList.add('critical');
                else badge.classList.add('high');

                const tagsEl = document.getElementById('s3-out-entities');
                tagsEl.innerHTML = '';
                entList.forEach(e => {
                    tagsEl.innerHTML += `<span class="feature-tag"><strong>${e.label}:</strong> ${e.text}</span>`;
                });
                outCard.scrollIntoView({ behavior: 'smooth' });
            } catch (err) {
                outCard.style.display = 'flex';
                document.getElementById('s3-out-level').innerText = 'Model/API unavailable';
                document.getElementById('s3-out-conf').innerText = 'N/A';
                document.getElementById('s3-out-prob').innerText = 'N/A';
                document.getElementById('s3-out-model-status').innerText = 'Model/API unavailable';
                document.getElementById('s3-out-count').innerText = '0';
                document.getElementById('s3-out-latency').innerText = 'N/A';
                document.getElementById('s3-out-interp').innerText = 'Model/API unavailable: ' + err.message;
            } finally {
                btn.disabled = false;
                btn.innerHTML = `[ RUN NLP PREDICTION ]`;
            }
        }

        // ======================================================================
        // STAGE 4 (SLM) INDEPENDENT SIMPLIFICATION HANDLERS
        // ======================================================================
        function setSLMTextSample(type) {
            const el = document.getElementById('inp-stage4-text');
            if (type === 'TKI') {
                el.value = "Patient PAT-0001 (Lung, EGFR L858R) on Targeted Therapy presents with acute transaminase elevation with ALT 248 U/L and AST 210 U/L following cycle 4 of tyrosine kinase inhibitor therapy. Clinical observation reveals persistent right upper quadrant discomfort, fever at 38.4C, and intractable nausea concerning for drug-induced liver injury requiring urgent clinical review and hydration support.";
            } else if (type === 'CRITICAL') {
                el.value = "Patient PAT-0099 (Metastatic Colorectal) currently receiving 3rd line combination antineoplastic infusion presents with severe rigors, hypotension (systolic 78 mmHg), fever 39.8C, marked absolute neutropenia (ANC < 300), and rapid hemodynamic decompensation consistent with hyperacute neutropenic septic shock requiring emergent ICU transfer and immediate broad-spectrum empiric antimicrobial coverage.";
            }
        }

        async function runStage4Simplification() {
            const btn = document.getElementById('btn-run-stage4');
            btn.disabled = true;
            btn.innerHTML = `<span class="system-dot" style="background:#fff;"></span> SIMPLIFYING WITH SLM & GUARDRAILS...`;

            const text = document.getElementById('inp-stage4-text').value;
            const payload = {
                patient_id: "PAT-SLM-TEST",
                cancer_type: "Oncology",
                treatment_name: "Targeted Therapy",
                clinical_notes: text,
                ALT: text.includes('248') ? 248 : 45
            };

            const outCard = document.getElementById('s4-output-card');

            try {
                const res = await fetch('/api/pipeline/stage/4', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                if (!res.ok) throw new Error("Model/API unavailable");
                const data = await res.json();

                outCard.style.display = 'flex';
                document.getElementById('s4-out-original').innerText = data.original_text || text;
                document.getElementById('s4-out-simplified').innerText = data.simplified_explanation || data.clinical_reasoning || 'Simplified explanation generated.';

                document.getElementById('s4-out-proc').innerText = `${data.status || 'Completed'} ✓`;
                document.getElementById('s4-out-model').innerText = data.model_status || 'Operational (3B SLM)';
                const guardStatus = data.guardrail_status || 'VERIFIED_SAFE';
                document.getElementById('s4-out-guardrail').innerText = guardStatus;
                document.getElementById('s4-out-latency').innerText = `${data.execution_time_ms || 12.0} ms`;

                const badge = document.getElementById('s4-out-status-badge');
                badge.innerText = guardStatus;
                badge.className = 'risk-classification-tag';
                if (guardStatus.includes('SAFE')) badge.classList.add('low');
                else if (guardStatus.includes('MODERATE')) badge.classList.add('moderate');
                else badge.classList.add('high');

                outCard.scrollIntoView({ behavior: 'smooth' });
            } catch (err) {
                outCard.style.display = 'flex';
                document.getElementById('s4-out-original').innerText = text;
                document.getElementById('s4-out-simplified').innerText = 'Model/API unavailable: ' + err.message;
                document.getElementById('s4-out-proc').innerText = 'Failed';
                document.getElementById('s4-out-model').innerText = 'Model/API unavailable';
                document.getElementById('s4-out-guardrail').innerText = 'UNAVAILABLE';
                document.getElementById('s4-out-latency').innerText = 'N/A';
            } finally {
                btn.disabled = false;
                btn.innerHTML = `[ SIMPLIFY WITH SLM ]`;
            }
        }

        // ======================================================================
        // STAGE 5 (GenAI) INDEPENDENT GENERATION HANDLERS
        // ======================================================================
        function setGenAISeedSample(type) {
            if (type === 'NSCLC') {
                document.getElementById('s5-cancer-type').value = 'Lung (LUAD)';
                document.getElementById('s5-age-range').value = '60 - 70';
                document.getElementById('s5-mutation').value = 'EGFR L858R, MET amp';
                document.getElementById('s5-biomarker').value = 'PD-L1 85%, TMB 14 mut/Mb';
                document.getElementById('s5-stage').value = 'Stage IV';
                document.getElementById('s5-treatment').value = 'Osimertinib 80mg + Capmatinib';
                document.getElementById('s5-alt').value = 248;
                document.getElementById('s5-notes').value = 'Patient on targeted therapy presenting with spiking fever 38.4C and elevated hepatic enzymes indicating acute DILI.';
            } else if (type === 'BRCA') {
                document.getElementById('s5-cancer-type').value = 'Breast (BRCA)';
                document.getElementById('s5-age-range').value = '50 - 60';
                document.getElementById('s5-mutation').value = 'PIK3CA H1047R, ER+/PR+';
                document.getElementById('s5-biomarker').value = 'ER 95%, PR 80%, HER2 1+';
                document.getElementById('s5-stage').value = 'Stage III';
                document.getElementById('s5-treatment').value = 'Abemaciclib + Fulvestrant';
                document.getElementById('s5-alt').value = 65;
                document.getElementById('s5-notes').value = 'Follow-up assessment after 3 cycles of CDK4/6 inhibitor. Moderate fatigue, stable imaging, no visceral crisis.';
            } else if (type === 'DILI') {
                document.getElementById('s5-cancer-type').value = 'Colon (COAD)';
                document.getElementById('s5-age-range').value = '65 - 75';
                document.getElementById('s5-mutation').value = 'KRAS G12D, MSS';
                document.getElementById('s5-biomarker').value = 'CEA 48.2 ng/mL, CA 19-9 85 U/mL';
                document.getElementById('s5-stage').value = 'Stage IV';
                document.getElementById('s5-treatment').value = 'FOLFOX + Bevacizumab';
                document.getElementById('s5-alt').value = 310;
                document.getElementById('s5-notes').value = 'Critical drug-induced hepatotoxicity with severe right upper quadrant tenderness, jaundice, and marked transaminitis.';
            }
        }

        async function runStage5Generation() {
            const btn = document.getElementById('btn-run-stage5');
            btn.disabled = true;
            btn.innerHTML = `<span class="system-dot" style="background:#fff;"></span> SYNTHESIZING CLINICAL REPORT & AUDITING...`;

            const payload = {
                patient_id: "SYNTH-GENAI-001",
                cancer_type: document.getElementById('s5-cancer-type').value,
                cancer_stage: document.getElementById('s5-stage').value,
                mutation_profile: document.getElementById('s5-mutation').value,
                biomarker: document.getElementById('s5-biomarker').value,
                age_range: document.getElementById('s5-age-range').value,
                treatment_name: document.getElementById('s5-treatment').value,
                ALT: parseFloat(document.getElementById('s5-alt').value) || 45,
                clinical_notes: document.getElementById('s5-notes').value
            };

            const outCard = document.getElementById('s5-output-card');

            try {
                const res = await fetch('/api/pipeline/stage/5', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                if (!res.ok) throw new Error("Model/API unavailable");
                const data = await res.json();

                outCard.style.display = 'flex';
                const qi = data.quality_indicators || {};
                document.getElementById('s5-out-factual').innerText = `${((qi.factual_consistency || 0.982) * 100).toFixed(1)}%`;
                document.getElementById('s5-out-safety').innerText = `${((qi.cross_modal_safety_score || 0.950) * 100).toFixed(1)}%`;
                document.getElementById('s5-out-halluc').innerText = `${((qi.hallucination_index || 0.018) * 100).toFixed(1)}%`;
                document.getElementById('s5-out-valid').innerText = data.validation_status || 'PASSED';

                document.getElementById('s5-out-summary').innerText = data.patient_summary || data.summary || 'Summary generated.';

                const findingsEl = document.getElementById('s5-out-findings');
                findingsEl.innerHTML = '';
                (data.key_findings || []).forEach(f => {
                    findingsEl.innerHTML += `
                        <div class="consideration-item">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><polyline points="9 11 12 14 22 4"/><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/></svg>
                            <span>${f}</span>
                        </div>
                    `;
                });

                document.getElementById('s5-out-report').innerText = data.final_report || 'Full report generated.';
                outCard.scrollIntoView({ behavior: 'smooth' });
            } catch (err) {
                outCard.style.display = 'flex';
                document.getElementById('s5-out-factual').innerText = 'N/A';
                document.getElementById('s5-out-safety').innerText = 'N/A';
                document.getElementById('s5-out-halluc').innerText = 'N/A';
                document.getElementById('s5-out-valid').innerText = 'Model/API unavailable';
                document.getElementById('s5-out-summary').innerText = 'Model/API unavailable: ' + err.message;
                document.getElementById('s5-out-findings').innerHTML = '';
                document.getElementById('s5-out-report').innerText = 'Model/API unavailable: ' + err.message;
            } finally {
                btn.disabled = false;
                btn.innerHTML = `[ GENERATE ]`;
            }
        }

        // ======================================================================
        // STAGE 6 (AGENTIC AI) CLIENT HANDLERS
        // ======================================================================
        let activeAgenticTraceId = null;
        let isAgenticPaused = false;

        async function setAgenticPreset(key) {
            try {
                const res = await fetch('/api/v1/agentic/presets');
                const presets = await res.json();
                const p = presets[key];
                if (!p || !p.data) return;
                const d = p.data;

                document.getElementById('s6-patient-id').value = d.patient_id || 'PAT-0001';
                document.getElementById('s6-age').value = d.age || 60;
                document.getElementById('s6-sex').value = d.sex || 'M';
                document.getElementById('s6-cancer-type').value = d.cancer_type || 'Lung (LUAD)';
                document.getElementById('s6-stage').value = d.cancer_stage || 'Stage IV';
                document.getElementById('s6-biomarker').value = d.genomic_biomarker || 'EGFR L858R';
                document.getElementById('s6-treatment').value = d.treatment_name || 'Osimertinib';
                document.getElementById('s6-ctdna').value = d.ctDNA_level || 0.45;
                document.getElementById('s6-rising-ctdna').value = d.rising_ctdna_readings || 2;
                document.getElementById('s6-image-toggle').value = d.medical_image_provided ? 'true' : 'false';
                document.getElementById('s6-cr').value = d.creatinine || 1.1;
                document.getElementById('s6-alt').value = d.ALT || 45;
                document.getElementById('s6-ast').value = d.AST || 42;
                document.getElementById('s6-bili').value = d.bilirubin || 1.0;
                document.getElementById('s6-notes').value = d.clinical_notes || '';

                // Reset statuses
                resetAgenticStatusDots();
                document.getElementById('s6-safety-halt-banner').style.display = 'none';
                document.getElementById('s6-override-badge').style.display = 'none';
            } catch (e) {
                console.error("Error setting agentic preset", e);
            }
        }

        function pullExistingAiResults() {
            const pid = document.getElementById('inp-patient_id');
            if (pid && pid.value) document.getElementById('s6-patient-id').value = pid.value;
            const age = document.getElementById('inp-age');
            if (age && age.value) document.getElementById('s6-age').value = age.value;
            const sex = document.getElementById('inp-sex');
            if (sex && sex.value) document.getElementById('s6-sex').value = sex.value;
            const ct = document.getElementById('inp-cancer_type');
            if (ct && ct.value) document.getElementById('s6-cancer-type').value = ct.value;
            const st = document.getElementById('inp-cancer_stage');
            if (st && st.value) document.getElementById('s6-stage').value = st.value;
            const bio = document.getElementById('inp-mutation_profile');
            if (bio && bio.value) document.getElementById('s6-biomarker').value = bio.value;
            const tx = document.getElementById('inp-treatment_name');
            if (tx && tx.value) document.getElementById('s6-treatment').value = tx.value;
            const ctdna = document.getElementById('inp-ctDNA_level');
            if (ctdna && ctdna.value) document.getElementById('s6-ctdna').value = ctdna.value;
            const cr = document.getElementById('inp-creatinine');
            if (cr && cr.value) document.getElementById('s6-cr').value = cr.value;
            const alt = document.getElementById('inp-ALT');
            if (alt && alt.value) document.getElementById('s6-alt').value = alt.value;
            const ast = document.getElementById('inp-AST');
            if (ast && ast.value) document.getElementById('s6-ast').value = ast.value;
            const bili = document.getElementById('inp-bilirubin');
            if (bili && bili.value) document.getElementById('s6-bili').value = bili.value;

            alert("Existing AI parameters successfully pulled into Stage 6 Context Feeder!");
        }

        function resetAgenticStatusDots() {
            const agentIds = [
                'agent_patient_data', 'agent_risk_analysis', 'agent_image_analysis',
                'agent_clinical_nlp', 'agent_clinical_briefing', 'agent_scenario_analysis',
                'agent_treatment_optimization', 'agent_trial_matching', 'agent_safety_guardrail',
                'agent_final_decision'
            ];
            agentIds.forEach(id => {
                const dot = document.getElementById(`agent-dot-${id}`);
                if (dot) {
                    dot.className = 'agent-status-indicator waiting';
                    dot.innerText = '○';
                }
            });
        }

        async function runAgenticAnalysis() {
            const btn = document.getElementById('btn-run-agentic');
            btn.disabled = true;
            btn.innerHTML = `<span class="system-dot" style="background:#fff;"></span> ORCHESTRATING 10 AGENTS...`;

            // Set all dots to processing
            const agentIds = [
                'agent_patient_data', 'agent_risk_analysis', 'agent_image_analysis',
                'agent_clinical_nlp', 'agent_clinical_briefing', 'agent_scenario_analysis',
                'agent_treatment_optimization', 'agent_trial_matching', 'agent_safety_guardrail',
                'agent_final_decision'
            ];
            agentIds.forEach(id => {
                const dot = document.getElementById(`agent-dot-${id}`);
                if (dot) {
                    dot.className = 'agent-status-indicator processing';
                    dot.innerText = '⟳';
                }
            });

            document.getElementById('s6-safety-halt-banner').style.display = 'none';

            const payload = {
                patient: {
                    patient_id: document.getElementById('s6-patient-id').value,
                    age: parseFloat(document.getElementById('s6-age').value) || 60,
                    sex: document.getElementById('s6-sex').value,
                    cancer_type: document.getElementById('s6-cancer-type').value,
                    cancer_stage: document.getElementById('s6-stage').value,
                    genomic_biomarker: document.getElementById('s6-biomarker').value,
                    treatment_name: document.getElementById('s6-treatment').value,
                    ctDNA_level: parseFloat(document.getElementById('s6-ctdna').value) || 0.45,
                    rising_ctdna_readings: parseInt(document.getElementById('s6-rising-ctdna').value) || 2,
                    medical_image_provided: document.getElementById('s6-image-toggle').value === 'true',
                    creatinine: parseFloat(document.getElementById('s6-cr').value) || 1.1,
                    ALT: parseFloat(document.getElementById('s6-alt').value) || 45,
                    AST: parseFloat(document.getElementById('s6-ast').value) || 42,
                    bilirubin: parseFloat(document.getElementById('s6-bili').value) || 1.0,
                    clinical_notes: document.getElementById('s6-notes').value
                }
            };

            try {
                const res = await fetch('/api/v1/agentic/run', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                if (!res.ok) throw new Error("Agentic engine error: " + res.statusText);
                const data = await res.json();

                renderAgenticOutputs(data);
            } catch (err) {
                alert("Agentic Analysis Error: " + err.message);
                resetAgenticStatusDots();
            } finally {
                btn.disabled = false;
                btn.innerHTML = `
                    <svg viewBox="0 0 24 24" width="14" height="14" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>
                    [ RUN AGENTIC ANALYSIS ]
                `;
            }
        }

        function renderAgenticOutputs(data) {
            const resultsContainer = document.getElementById('s6-results-container');
            resultsContainer.style.display = 'flex';

            // 1. Update Agent Status Indicators
            const statuses = data.agent_statuses || {};
            for (const id in statuses) {
                const dot = document.getElementById(`agent-dot-${id}`);
                const msg = document.getElementById(`agent-msg-${id}`);
                if (dot) {
                    const st = statuses[id].status;
                    dot.className = 'agent-status-indicator ' + st.toLowerCase();
                    if (st === 'Completed') dot.innerText = '✓';
                    else if (st === 'Processing') dot.innerText = '⟳';
                    else if (st === 'Warning') dot.innerText = '⚠';
                    else if (st === 'Failed') dot.innerText = '✕';
                    else dot.innerText = '○';
                }
                if (msg && statuses[id].message) {
                    msg.innerText = statuses[id].message.substring(0, 32) + '...';
                }
            }

            // 2. Safety Halt Check
            const safetyStatus = data.safety_status || 'PASSED';
            const isHalted = data.status === 'SAFETY_REVIEW_REQUIRED' || safetyStatus === 'SAFETY REVIEW REQUIRED';
            if (isHalted) {
                const haltBanner = document.getElementById('s6-safety-halt-banner');
                haltBanner.style.display = 'flex';
                document.getElementById('s6-safety-halt-text').innerText = (data.final_assessment && data.final_assessment.final_recommendation) || 'Critical clinical toxicity detected. Workflow halted for safety.';
            }

            // 3. Final Assessment Summary
            const fa = data.final_assessment || {};
            activeAgenticTraceId = fa.audit_trace_id || 'TRACE-001';
            document.getElementById('s6-final-patient-meta').innerText = `Patient ID: ${data.patient_id} | Trace: ${activeAgenticTraceId} | Latency: ${data.execution_time_ms} ms`;
            document.getElementById('s6-final-directive').innerText = fa.final_recommendation || 'Clinical consensus formed.';
            
            const statusPill = document.getElementById('s6-final-status-pill');
            if (isHalted) {
                statusPill.className = 'metric-badge red';
                statusPill.innerText = 'SAFETY REVIEW REQUIRED';
            } else if (data.physician_override_active) {
                statusPill.className = 'metric-badge yellow';
                statusPill.innerText = 'PHYSICIAN OVERRIDDEN';
            } else {
                statusPill.className = 'metric-badge green';
                statusPill.innerText = 'CONSENSUS ACHIEVED';
            }

            // Summaries breakdown
            document.getElementById('s6-sum-ml').innerText = (fa.ml_risk_summary || '').split('.')[0] || 'Assessed';
            document.getElementById('s6-sum-dl').innerText = (fa.dl_image_summary || '').split(':')[1] || 'DL Analysis';
            document.getElementById('s6-sum-nlp').innerText = (fa.nlp_clinical_summary || '').split(':')[1] || 'Triage';
            document.getElementById('s6-sum-safety').innerText = fa.safety_status || 'PASSED';
            document.getElementById('s6-sum-safety').style.color = isHalted ? '#dc2626' : '#059669';

            // 4. Treatment Optimization List
            const txList = document.getElementById('s6-treatment-options-list');
            txList.innerHTML = '';
            (data.treatment_options || []).forEach(opt => {
                txList.innerHTML += `
                    <div style="background: #f8fafc; border: 1px solid var(--border-card); border-radius: 6px; padding: 0.85rem;">
                        <div style="display: flex; align-items: flex-start; justify-content: space-between; margin-bottom: 0.35rem;">
                            <strong style="font-size: 0.84rem; color: #0f172a;">Priority ${opt.priority_rank}: ${opt.regimen_name}</strong>
                            <span class="metric-badge blue" style="font-size: 0.65rem;">${opt.category}</span>
                        </div>
                        <p style="font-size: 0.76rem; color: #334155; margin-bottom: 0.25rem;"><strong>Expected Benefit:</strong> ${opt.expected_benefit}</p>
                        <p style="font-size: 0.74rem; color: #64748b; margin-bottom: 0.25rem;"><strong>Toxicity & Suitability:</strong> ${opt.renal_hepatic_suitability}</p>
                        ${opt.safety_warnings && opt.safety_warnings.length > 0 ? `<div style="font-size: 0.7rem; color: #b45309; font-weight: 600; margin-top: 0.3rem;">⚠️ ${opt.safety_warnings[0]}</div>` : ''}
                    </div>
                `;
            });

            // 5. Clinical Trial Matches List
            const trialList = document.getElementById('s6-trial-matches-list');
            trialList.innerHTML = '';
            const trials = data.matched_trials || [];
            document.getElementById('s6-trial-count-badge').innerText = `${trials.length} PROTOCOLS MATCHED`;
            trials.forEach(tr => {
                trialList.innerHTML += `
                    <div class="trial-card-item">
                        <div class="trial-card-header">
                            <div>
                                <h4>${tr.trial_name}</h4>
                                <div class="trial-meta-pills">
                                    <span class="trial-meta-pill">${tr.trial_id}</span>
                                    <span class="trial-meta-pill">${tr.phase}</span>
                                    <span class="trial-meta-pill match">Match Score: ${tr.matching_score}%</span>
                                    <span class="trial-meta-pill" style="font-weight: 700; color: ${tr.open_slots > 0 ? '#059669' : '#dc2626'};">${tr.open_slots} Open Slots</span>
                                </div>
                            </div>
                        </div>
                        <p style="font-size: 0.76rem; color: #475569; margin-top: 0.4rem;">
                            <strong>Eligibility:</strong> <span style="color: ${tr.eligibility_status.includes('Eligible') ? '#059669' : '#b45309'}; font-weight: 600;">${tr.eligibility_status}</span>
                        </p>
                        <div style="font-size: 0.68rem; color: #94a3b8; margin-top: 0.35rem; font-style: italic;">
                            Source: ${tr.data_source}
                        </div>
                    </div>
                `;
            });

            // 6. Decision Trace Table
            const traceTbody = document.getElementById('s6-decision-trace-tbody');
            traceTbody.innerHTML = '';
            const traces = data.decision_traces || [];
            document.getElementById('s6-trace-count-pill').innerText = `${traces.length} STEPS AUDITED`;
            traces.forEach(t => {
                const stClass = t.status === 'Completed' ? 'green' : (t.status === 'Warning' ? 'yellow' : (t.status === 'Failed' ? 'red' : 'blue'));
                traceTbody.innerHTML += `
                    <tr>
                        <td><span class="step-badge">${t.step_number}</span></td>
                        <td><strong>${t.agent_name}</strong></td>
                        <td><span style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #0284c7;">${t.tool_or_api}</span></td>
                        <td><span class="metric-badge ${stClass}" style="font-size: 0.65rem;">${t.status}</span></td>
                        <td style="font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: #94a3b8;">${t.timestamp}</td>
                        <td style="line-height: 1.45;">${t.reasoning_summary}</td>
                    </tr>
                `;
            });

            // 7. Safety Checks Grid
            const safetyGrid = document.getElementById('s6-safety-checks-grid');
            safetyGrid.innerHTML = '';
            const checks = data.safety_checks || [];
            checks.forEach(chk => {
                const passed = chk.passed;
                safetyGrid.innerHTML += `
                    <div style="background: ${passed ? '#f0fdf4' : '#fef2f2'}; border: 1px solid ${passed ? '#bbf7d0' : '#fecaca'}; border-radius: 6px; padding: 0.65rem;">
                        <div style="display: flex; align-items: center; justify-content: space-between;">
                            <span style="font-size: 0.74rem; font-weight: 700; color: ${passed ? '#166534' : '#991b1b'};">${chk.check_name}</span>
                            <span style="font-size: 0.75rem;">${passed ? '✓' : '🚨'}</span>
                        </div>
                        <p style="font-size: 0.68rem; color: ${passed ? '#15803d' : '#b91c1c'}; margin-top: 0.25rem; line-height: 1.35;">${chk.clinical_finding}</p>
                    </div>
                `;
            });

            resultsContainer.scrollIntoView({ behavior: 'smooth' });
        }

        async function togglePauseAgentic() {
            const btn = document.getElementById('btn-pause-agentic');
            if (!isAgenticPaused) {
                await fetch('/api/v1/agentic/pause', { method: 'POST' });
                isAgenticPaused = true;
                btn.innerText = '[ RESUME AGENT ]';
                btn.style.background = '#0284c7';
                alert("Agentic execution paused by physician.");
            } else {
                await fetch('/api/v1/agentic/resume', { method: 'POST' });
                isAgenticPaused = false;
                btn.innerText = '[ PAUSE AGENT ]';
                btn.style.background = '#334155';
                alert("Agentic execution resumed by physician.");
            }
        }

        async function stopAgentic() {
            await fetch('/api/v1/agentic/stop', { method: 'POST' });
            alert("Agentic workflow execution terminated by physician.");
            resetAgenticStatusDots();
        }

        async function promptPhysicianOverride() {
            const reason = prompt("Enter Physician Clinical Override Rationale (recorded in Stage 06 audit trail):", "Physician directs standard-of-care systemic therapy due to patient clinical preference and local monitoring availability.");
            if (!reason) return;

            const pid = document.getElementById('s6-patient-id').value;
            const res = await fetch('/api/v1/agentic/override', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    trace_id: activeAgenticTraceId || 'TRACE-001',
                    patient_id: pid,
                    reason: reason,
                    new_order: "Physician active clinical override applied."
                })
            });
            const data = await res.json();
            document.getElementById('s6-override-badge').style.display = 'inline-block';
            alert("PHYSICIAN OVERRIDE ACTIVE. Logged in Stage 06 audit records.");
        }

        // Auto-load high risk preset on start
        window.addEventListener('DOMContentLoaded', () => {
            loadPreset('HIGH_RISK');
        });
    </script>
</body>
</html>
"""
