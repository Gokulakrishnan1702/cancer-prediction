"""
Canonical PatientContext Schema
Structured, validated object representation for oncology patient context.
Supports both structured sub-models (demographics, cancer_information, biomarkers, etc.)
and flat field access (e.g., patient.age, patient.creatinine) for seamless backward compatibility.
"""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class Demographics(BaseModel):
    age: float = 62.0
    sex: str = "M"


class CancerInformation(BaseModel):
    cancer_type: str = "Lung (LUAD)"
    cancer_stage: str = "Stage IV"
    histology: Optional[str] = "Adenocarcinoma"


class Biomarkers(BaseModel):
    genomic_biomarker: str = "EGFR L858R"
    biomarker_status: str = "Positive"
    tumor_marker_level: float = 18.5
    ctDNA_level: float = 0.85
    ctDNA_trajectory: List[float] = Field(default_factory=lambda: [0.15, 0.28, 0.42, 0.58, 0.72, 0.85])
    rising_ctdna_readings: int = 6


class Mutations(BaseModel):
    mutation_profile: str = "EGFR L858R + TP53"
    mutation_count: int = 2
    variants: List[str] = Field(default_factory=lambda: ["EGFR:c.2573T>G:p.Leu858Arg", "TP53:c.524G>A:p.Arg175His"])


class LaboratoryResults(BaseModel):
    WBC: float = 6.8
    hemoglobin: float = 12.2
    platelet_count: float = 230.0
    creatinine: float = 1.15
    bilirubin: float = 1.10
    ALT: float = 48.0
    AST: float = 44.0
    systolic_bp: float = 128.0
    diastolic_bp: float = 82.0
    heart_rate: float = 78.0
    spo2: float = 97.0


class TreatmentHistory(BaseModel):
    treatment_name: str = "Osimertinib (Targeted TKI)"
    treatment_cycle: int = 4
    dosage_mg: float = 80.0
    previous_dose_mg: float = 80.0
    treatment_duration_days: float = 84.0
    prior_therapies: str = "Prior 1st-line chemotherapy (Carboplatin + Pemetrexed) 6 cycles; currently on Osimertinib."
    treatment_response: str = "Initial PR, subsequent molecular progression"
    toxicity_adverse_events: str = "Grade 1 transaminitis, mild skin rash"


class ClinicalNotes(BaseModel):
    notes_text: str = (
        "62yo male with metastatic EGFR+ NSCLC. Exhibits rapid progression on imaging. "
        "6 consecutive rising ctDNA timepoints indicating emergence of resistance. "
        "Tolerates Osimertinib but transaminases elevated."
    )
    extracted_symptoms: List[str] = Field(default_factory=lambda: ["cough", "mild dyspnea", "fatigue"])
    extracted_drugs: List[str] = Field(default_factory=lambda: ["Osimertinib", "Carboplatin", "Pemetrexed"])


class ImageInformation(BaseModel):
    medical_image_provided: bool = True
    image_file_path: Optional[str] = "/api/pipeline/image/lung/lung_001.png"
    modalities: List[str] = Field(default_factory=lambda: ["CT Chest", "PET/CT"])


class PatientContext(BaseModel):
    patient_id: str = "PAT-0001"
    demographics: Demographics = Field(default_factory=Demographics)
    cancer_information: CancerInformation = Field(default_factory=CancerInformation)
    biomarkers: Biomarkers = Field(default_factory=Biomarkers)
    mutations: Mutations = Field(default_factory=Mutations)
    laboratory_results: LaboratoryResults = Field(default_factory=LaboratoryResults)
    treatment_history: TreatmentHistory = Field(default_factory=TreatmentHistory)
    clinical_notes: str = (
        "62yo male with metastatic EGFR+ NSCLC. Exhibits rapid progression on imaging. "
        "6 consecutive rising ctDNA timepoints indicating emergence of resistance. "
        "Tolerates Osimertinib but transaminases elevated."
    )
    image_information: ImageInformation = Field(default_factory=ImageInformation)
    previous_ai_results: Optional[Dict[str, Any]] = None

    def __init__(self, **data):
        # Extract flat properties if passed directly into init
        demo_args = {}
        if "age" in data:
            demo_args["age"] = float(data.pop("age"))
        if "sex" in data:
            demo_args["sex"] = str(data.pop("sex"))

        cancer_args = {}
        if "cancer_type" in data:
            cancer_args["cancer_type"] = str(data.pop("cancer_type"))
        if "cancer_stage" in data:
            cancer_args["cancer_stage"] = str(data.pop("cancer_stage"))

        bio_args = {}
        for k in ["genomic_biomarker", "biomarker_status", "tumor_marker_level", "ctDNA_level", "ctDNA_trajectory", "rising_ctdna_readings"]:
            if k in data:
                bio_args[k] = data.pop(k)

        mut_args = {}
        for k in ["mutation_profile", "mutation_count", "variants"]:
            if k in data:
                mut_args[k] = data.pop(k)

        lab_args = {}
        for k in ["WBC", "hemoglobin", "platelet_count", "creatinine", "bilirubin", "ALT", "AST", "systolic_bp", "diastolic_bp", "heart_rate", "spo2"]:
            if k in data:
                lab_args[k] = float(data.pop(k))

        tx_args = {}
        for k in ["treatment_name", "treatment_cycle", "dosage_mg", "previous_dose_mg", "treatment_duration_days", "prior_therapies", "treatment_response", "toxicity_adverse_events"]:
            if k in data:
                tx_args[k] = data.pop(k)

        img_args = {}
        if "medical_image_provided" in data:
            img_args["medical_image_provided"] = bool(data.pop("medical_image_provided"))
        if "image_file_path" in data:
            img_args["image_file_path"] = data.pop("image_file_path")

        if demo_args:
            data["demographics"] = Demographics(**demo_args)
        if cancer_args:
            data["cancer_information"] = CancerInformation(**cancer_args)
        if bio_args:
            data["biomarkers"] = Biomarkers(**bio_args)
        if mut_args:
            data["mutations"] = Mutations(**mut_args)
        if lab_args:
            data["laboratory_results"] = LaboratoryResults(**lab_args)
        if tx_args:
            data["treatment_history"] = TreatmentHistory(**tx_args)
        if img_args:
            data["image_information"] = ImageInformation(**img_args)

        if "clinical_notes" not in data or not data["clinical_notes"]:
            c_type_val = cancer_args.get("cancer_type", "Lung (LUAD)")
            c_stage_val = cancer_args.get("cancer_stage", "Stage IV")
            bio_val = bio_args.get("genomic_biomarker", "EGFR L858R")
            pid_val = data.get("patient_id", "PAT-0001")
            data["clinical_notes"] = f"Clinical oncology case notes for patient {pid_val} diagnosed with {c_type_val} ({c_stage_val}) harboring {bio_val}."

        super().__init__(**data)

    # -------------------------------------------------------------
    # Transparent Flat Properties for Seamless Codebase Compatibility
    # -------------------------------------------------------------
    @property
    def age(self) -> float:
        return self.demographics.age

    @age.setter
    def age(self, value: float):
        self.demographics.age = float(value)

    @property
    def sex(self) -> str:
        return self.demographics.sex

    @sex.setter
    def sex(self, value: str):
        self.demographics.sex = str(value)

    @property
    def cancer_type(self) -> str:
        return self.cancer_information.cancer_type

    @cancer_type.setter
    def cancer_type(self, value: str):
        self.cancer_information.cancer_type = str(value)

    @property
    def cancer_stage(self) -> str:
        return self.cancer_information.cancer_stage

    @cancer_stage.setter
    def cancer_stage(self, value: str):
        self.cancer_information.cancer_stage = str(value)

    @property
    def genomic_biomarker(self) -> str:
        return self.biomarkers.genomic_biomarker

    @genomic_biomarker.setter
    def genomic_biomarker(self, value: str):
        self.biomarkers.genomic_biomarker = str(value)

    @property
    def biomarker_status(self) -> str:
        return self.biomarkers.biomarker_status

    @biomarker_status.setter
    def biomarker_status(self, value: str):
        self.biomarkers.biomarker_status = str(value)

    @property
    def tumor_marker_level(self) -> float:
        return self.biomarkers.tumor_marker_level

    @tumor_marker_level.setter
    def tumor_marker_level(self, value: float):
        self.biomarkers.tumor_marker_level = float(value)

    @property
    def ctDNA_level(self) -> float:
        return self.biomarkers.ctDNA_level

    @ctDNA_level.setter
    def ctDNA_level(self, value: float):
        self.biomarkers.ctDNA_level = float(value)

    @property
    def ctDNA_trajectory(self) -> List[float]:
        return self.biomarkers.ctDNA_trajectory

    @ctDNA_trajectory.setter
    def ctDNA_trajectory(self, value: List[float]):
        self.biomarkers.ctDNA_trajectory = value

    @property
    def rising_ctdna_readings(self) -> int:
        return self.biomarkers.rising_ctdna_readings

    @rising_ctdna_readings.setter
    def rising_ctdna_readings(self, value: int):
        self.biomarkers.rising_ctdna_readings = int(value)

    @property
    def mutation_profile(self) -> str:
        return self.mutations.mutation_profile

    @mutation_profile.setter
    def mutation_profile(self, value: str):
        self.mutations.mutation_profile = str(value)

    @property
    def mutation_count(self) -> int:
        return self.mutations.mutation_count

    @mutation_count.setter
    def mutation_count(self, value: int):
        self.mutations.mutation_count = int(value)

    @property
    def creatinine(self) -> float:
        return self.laboratory_results.creatinine

    @creatinine.setter
    def creatinine(self, value: float):
        self.laboratory_results.creatinine = float(value)

    @property
    def bilirubin(self) -> float:
        return self.laboratory_results.bilirubin

    @bilirubin.setter
    def bilirubin(self, value: float):
        self.laboratory_results.bilirubin = float(value)

    @property
    def ALT(self) -> float:
        return self.laboratory_results.ALT

    @ALT.setter
    def ALT(self, value: float):
        self.laboratory_results.ALT = float(value)

    @property
    def AST(self) -> float:
        return self.laboratory_results.AST

    @AST.setter
    def AST(self, value: float):
        self.laboratory_results.AST = float(value)

    @property
    def platelet_count(self) -> float:
        return self.laboratory_results.platelet_count

    @platelet_count.setter
    def platelet_count(self, value: float):
        self.laboratory_results.platelet_count = float(value)

    @property
    def WBC(self) -> float:
        return self.laboratory_results.WBC

    @WBC.setter
    def WBC(self, value: float):
        self.laboratory_results.WBC = float(value)

    @property
    def hemoglobin(self) -> float:
        return self.laboratory_results.hemoglobin

    @hemoglobin.setter
    def hemoglobin(self, value: float):
        self.laboratory_results.hemoglobin = float(value)

    @property
    def systolic_bp(self) -> float:
        return self.laboratory_results.systolic_bp

    @systolic_bp.setter
    def systolic_bp(self, value: float):
        self.laboratory_results.systolic_bp = float(value)

    @property
    def diastolic_bp(self) -> float:
        return self.laboratory_results.diastolic_bp

    @diastolic_bp.setter
    def diastolic_bp(self, value: float):
        self.laboratory_results.diastolic_bp = float(value)

    @property
    def heart_rate(self) -> float:
        return self.laboratory_results.heart_rate

    @heart_rate.setter
    def heart_rate(self, value: float):
        self.laboratory_results.heart_rate = float(value)

    @property
    def spo2(self) -> float:
        return self.laboratory_results.spo2

    @spo2.setter
    def spo2(self, value: float):
        self.laboratory_results.spo2 = float(value)

    @property
    def treatment_name(self) -> str:
        return self.treatment_history.treatment_name

    @treatment_name.setter
    def treatment_name(self, value: str):
        self.treatment_history.treatment_name = str(value)

    @property
    def treatment_cycle(self) -> int:
        return self.treatment_history.treatment_cycle

    @treatment_cycle.setter
    def treatment_cycle(self, value: int):
        self.treatment_history.treatment_cycle = int(value)

    @property
    def dosage_mg(self) -> float:
        return self.treatment_history.dosage_mg

    @dosage_mg.setter
    def dosage_mg(self, value: float):
        self.treatment_history.dosage_mg = float(value)

    @property
    def previous_dose_mg(self) -> float:
        return self.treatment_history.previous_dose_mg

    @previous_dose_mg.setter
    def previous_dose_mg(self, value: float):
        self.treatment_history.previous_dose_mg = float(value)

    @property
    def treatment_duration_days(self) -> float:
        return self.treatment_history.treatment_duration_days

    @treatment_duration_days.setter
    def treatment_duration_days(self, value: float):
        self.treatment_history.treatment_duration_days = float(value)

    @property
    def medical_image_provided(self) -> bool:
        return self.image_information.medical_image_provided

    @medical_image_provided.setter
    def medical_image_provided(self, value: bool):
        self.image_information.medical_image_provided = bool(value)

    @property
    def image_file_path(self) -> Optional[str]:
        return self.image_information.image_file_path

    @image_file_path.setter
    def image_file_path(self, value: Optional[str]):
        self.image_information.image_file_path = value
