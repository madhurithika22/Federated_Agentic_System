from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, List, Dict

# --- 1. Basic Definitions ---

class JobType(str, Enum):
    TRAINING = "training"
    EVALUATION = "evaluation"

class NegotiationStatus(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    COUNTER_OFFER = "counter_offer"

# --- 2. The Privacy Budget ---
# This is the core of our "Product". We treat Privacy as a currency.
class PrivacyBudget(BaseModel):
    epsilon: float = Field(..., description="Differential Privacy Epsilon (lower = more private)")
    delta: float = Field(1e-5, description="Probability of privacy breach (usually < 1/dataset_size)")
    clipping_norm: float = Field(1.0, description="Max gradient norm before clipping")

# --- 3. The Resource Description (Discovery) ---
# What the Agent shows to the world via MCP
class DataResourceProfile(BaseModel):
    agent_id: str
    data_size: int = Field(..., description="Number of samples available")
    features: List[str] = Field(..., description="List of available feature columns")
    label_distribution: Dict[str, int] = Field(..., description="Distribution of target classes for fairness checks")
    
    # Simple metric for agent to signal quality without revealing data
    data_freshness_score: float = Field(0.0, ge=0.0, le=1.0) 

# --- 4. The Negotiation Payloads (The "Handshake") ---

class TrainingProposal(BaseModel):
    job_id: str
    job_type: JobType
    privacy_budget: PrivacyBudget
    payment_offer: float = Field(..., description="Credits offered for this round")
    rounds: int = Field(1, description="Number of FL rounds required")

class NegotiationResponse(BaseModel):
    job_id: str
    status: NegotiationStatus
    reason: Optional[str] = None
    
    # If rejecting, the agent can suggest what it wants
    counter_proposal: Optional[Dict[str, float]] = Field(
        None, description="e.g. {'payment_offer': 20.0, 'epsilon': 0.5}"
    )

# --- 5. The Contract (Final Agreement) ---
class TrainingContract(BaseModel):
    job_id: str
    agent_id: str
    agreed_price: float
    agreed_privacy: PrivacyBudget
    digital_signature: str = Field(..., description="Mock signature validating the deal")

print("✅ Protocol Schemas Defined Successfully.")