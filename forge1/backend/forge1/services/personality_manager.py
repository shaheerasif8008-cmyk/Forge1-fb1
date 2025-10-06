# forge1/backend/forge1/services/personality_manager.py
"""
Employee Personality and Behavior Management System

Manages personality trait application, behavior configuration, and communication
style adaptation for AI employees.

Requirements: 4.1, 4.2, 4.3, 4.4, 4.5
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

from forge1.models.employee_models import (
    PersonalityConfig, ModelPreferences, Employee,
    CommunicationStyle, FormalityLevel, ExpertiseLevel, ResponseLength,
    InvalidConfigurationError
)

logger = logging.getLogger(__name__)


class PersonalityDimension(Enum):
    """Personality dimensions for behavior analysis"""
    COMMUNICATION_STYLE = "communication_style"
    FORMALITY_LEVEL = "formality_level"
    EXPERTISE_LEVEL = "expertise_level"
    RESPONSE_LENGTH = "response_length"
    CREATIVITY_LEVEL = "creativity_level"
    EMPATHY_LEVEL = "empathy_level"


class BehaviorContext(Enum):
    """Different contexts that may require behavior adaptation"""
    PROFESSIONAL_MEETING = "professional_meeting"
    CASUAL_CONVERSATION = "casual_conversation"
    TECHNICAL_DISCUSSION = "technical_discussion"
    CUSTOMER_SUPPORT = "customer_support"
    CRISIS_MANAGEMENT = "crisis_management"
    TRAINING_SESSION = "training_session"
    CREATIVE_BRAINSTORMING = "creative_brainstorming"


class PersonalityManager:
    """
    Manages AI employee personality traits and behavior adaptation.
    
    Features:
    - Personality trait configuration and validation
    - Behavior adaptation based on context
    - Communication style enforcement
    - Model preference optimization
    - Personality consistency checking
    """
    
    def __init__(self):
        self._personality_templates = self._initialize_personality_templates()
        self._behavior_rules = self._initialize_behavior_rules()
        self._communication_patterns = self._initialize_communication_patterns()
    
    def create_personality_from_requirements(
        self,
        role: str,
        industry: str,
        communication_style: CommunicationStyle,
        formality_level: FormalityLevel,
        expertise_level: ExpertiseLevel,
        response_length: ResponseLength,
        creativity_level: float,
        empathy_level: float,
        custom_traits: Optional[Dict[str, Any]] = None
    ) -> PersonalityConfig:
        """
        Create a personality configuration from requirements with validation and optimization.
        """
        try:
            # Validate numeric levels
            if not 0.0 <= creativity_level <= 1.0:
                raise InvalidConfigurationError("Creativity level must be between 0.0 and 1.0")
            if not 0.0 <= empathy_level <= 1.0:
                raise InvalidConfigurationError("Empathy level must be between 0.0 and 1.0")
            
            # Apply role-specific adjustments
            adjusted_traits = self._apply_role_adjustments(
                role, industry, creativity_level, empathy_level
            )
            
            # Validate personality consistency
            self._validate_personality_consistency(
                communication_style, formality_level, expertise_level, 
                adjusted_traits["creativity"], adjusted_traits["empathy"]
            )
            
            # Merge custom traits with role-specific traits
            final_traits = {}
            if custom_traits:
                final_traits.update(custom_traits)
            
            # Add role-specific traits
            role_traits = self._get_role_specific_traits(role, industry)
            final_traits.update(role_traits)
            
            personality = PersonalityConfig(
                communication_style=communication_style,
                formality_level=formality_level,
                expertise_level=expertise_level,
                response_length=response_length,
                creativity_level=adjusted_traits["creativity"],
                empathy_level=adjusted_traits["empathy"],
                custom_traits=final_traits
            )
            
            logger.info(f"Created personality for {role} in {industry} industry")
            return personality
            
        except Exception as e:
            logger.error(f"Failed to create personality: {e}")
            raise InvalidConfigurationError(f"Personality creation failed: {e}")
    
    def adapt_personality_for_context(
        self,
        base_personality: PersonalityConfig,
        context: BehaviorContext,
        interaction_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Adapt personality traits based on interaction context and history.
        
        Returns adapted personality parameters for use in AI generation.
        """
        try:
            # Start with base personality
            adapted_traits = {
                "communication_style": base_personality.communication_style.value,
                "formality_level": base_personality.formality_level.value,
                "expertise_level": base_personality.expertise_level.value,
                "response_length": base_personality.response_length.value,
                "creativity_level": base_personality.creativity_level,
                "empathy_level": base_personality.empathy_level,
                "custom_traits": base_personality.custom_traits.copy()
            }
            
            # Apply context-specific adaptations
            context_adaptations = self._get_context_adaptations(context)
            
            # Adjust creativity based on context
            if context in [BehaviorContext.CREATIVE_BRAINSTORMING]:
                adapted_traits["creativity_level"] = min(1.0, adapted_traits["creativity_level"] + 0.2)
            elif context in [BehaviorContext.CRISIS_MANAGEMENT, BehaviorContext.TECHNICAL_DISCUSSION]:
                adapted_traits["creativity_level"] = max(0.0, adapted_traits["creativity_level"] - 0.2)
            
            # Adjust empathy based on context
            if context in [BehaviorContext.CUSTOMER_SUPPORT, BehaviorContext.CRISIS_MANAGEMENT]:
                adapted_traits["empathy_level"] = min(1.0, adapted_traits["empathy_level"] + 0.3)
            elif context in [BehaviorContext.TECHNICAL_DISCUSSION]:
                adapted_traits["empathy_level"] = max(0.2, adapted_traits["empathy_level"] - 0.1)
            
            # Adjust formality based on context
            if context == BehaviorContext.CASUAL_CONVERSATION:
                adapted_traits["formality_level"] = "casual"
            elif context in [BehaviorContext.PROFESSIONAL_MEETING, BehaviorContext.CRISIS_MANAGEMENT]:
                adapted_traits["formality_level"] = "formal"
            
            # Adjust response length based on context
            if context == BehaviorContext.CRISIS_MANAGEMENT:
                adapted_traits["response_length"] = "concise"
            elif context in [BehaviorContext.TRAINING_SESSION, BehaviorContext.TECHNICAL_DISCUSSION]:
                adapted_traits["response_length"] = "detailed"
            
            # Apply interaction history learning
            if interaction_history:
                history_adaptations = self._learn_from_interaction_history(
                    interaction_history, adapted_traits
                )
                adapted_traits.update(history_adaptations)
            
            # Add context-specific behavioral instructions
            adapted_traits["context_instructions"] = self._generate_context_instructions(
                context, adapted_traits
            )
            
            logger.debug(f"Adapted personality for context: {context.value}")
            return adapted_traits
            
        except Exception as e:
            logger.error(f"Failed to adapt personality for context {context}: {e}")
            # Return base personality as fallback
            return {
                "communication_style": base_personality.communication_style.value,
                "formality_level": base_personality.formality_level.value,
                "expertise_level": base_personality.expertise_level.value,
                "response_length": base_personality.response_length.value,
                "creativity_level": base_personality.creativity_level,
                "empathy_level": base_personality.empathy_level,
                "custom_traits": base_personality.custom_traits,
                "context_instructions": []
            }
    
    def optimize_model_preferences_for_personality(
        self,
        personality: PersonalityConfig,
        base_preferences: ModelPreferences,
        task_type: Optional[str] = None
    ) -> ModelPreferences:
        """
        Optimize model preferences based on personality traits and task requirements.
        """
        try:
            # Start with base preferences
            optimized = ModelPreferences(
                primary_model=base_preferences.primary_model,
                fallback_models=base_preferences.fallback_models.copy(),
                temperature=base_preferences.temperature,
                max_tokens=base_preferences.max_tokens,
                specialized_models=base_preferences.specialized_models.copy()
            )
            
            # Adjust temperature based on creativity level
            creativity_adjustment = (personality.creativity_level - 0.5) * 0.4  # -0.2 to +0.2
            optimized.temperature = max(0.0, min(2.0, 
                base_preferences.temperature + creativity_adjustment
            ))
            
            # Adjust max_tokens based on response length preference
            if personality.response_length == ResponseLength.CONCISE:
                optimized.max_tokens = min(optimized.max_tokens, 1000)
            elif personality.response_length == ResponseLength.DETAILED:
                optimized.max_tokens = max(optimized.max_tokens, 2000)
            
            # Adjust model selection based on expertise level
            if personality.expertise_level == ExpertiseLevel.EXPERT:
                # Prefer more capable models for expert-level responses
                if "gpt-3.5-turbo" in optimized.primary_model:
                    optimized.primary_model = "gpt-4"
            
            # Add task-specific model preferences
            if task_type:
                task_model = self._get_optimal_model_for_task(
                    task_type, personality, optimized.primary_model
                )
                if task_model and task_model != optimized.primary_model:
                    optimized.specialized_models[task_type] = task_model
            
            logger.debug(f"Optimized model preferences for personality")
            return optimized
            
        except Exception as e:
            logger.error(f"Failed to optimize model preferences: {e}")
            return base_preferences
    
    def validate_personality_consistency(self, personality: PersonalityConfig) -> List[str]:
        """
        Validate personality configuration for consistency and provide warnings.
        
        Returns list of validation warnings (empty if all good).
        """
        warnings = []
        
        try:
            # Check for conflicting traits
            if (personality.communication_style == CommunicationStyle.TECHNICAL and 
                personality.empathy_level > 0.8):
                warnings.append("High empathy level may conflict with technical communication style")
            
            if (personality.formality_level == FormalityLevel.CASUAL and 
                personality.expertise_level == ExpertiseLevel.EXPERT):
                warnings.append("Casual formality may not suit expert-level expertise")
            
            if (personality.response_length == ResponseLength.CONCISE and 
                personality.creativity_level > 0.8):
                warnings.append("High creativity may conflict with concise response preference")
            
            # Check custom traits for conflicts
            custom_traits = personality.custom_traits
            if custom_traits.get("attention_to_detail") == "high" and personality.creativity_level > 0.8:
                warnings.append("High creativity may conflict with high attention to detail")
            
            if custom_traits.get("risk_tolerance") == "low" and personality.creativity_level > 0.7:
                warnings.append("High creativity may conflict with low risk tolerance")
            
            # Check for extreme combinations
            if personality.creativity_level < 0.2 and personality.empathy_level < 0.2:
                warnings.append("Very low creativity and empathy may result in robotic responses")
            
            if personality.creativity_level > 0.9 and personality.empathy_level > 0.9:
                warnings.append("Very high creativity and empathy may result in unfocused responses")
            
        except Exception as e:
            logger.error(f"Personality validation failed: {e}")
            warnings.append(f"Validation error: {e}")
        
        return warnings
    
    def generate_personality_summary(self, personality: PersonalityConfig) -> str:
        """
        Generate a human-readable summary of the personality configuration.
        """
        try:
            summary_parts = []
            
            # Communication style
            style_descriptions = {
                CommunicationStyle.PROFESSIONAL: "maintains professional demeanor",
                CommunicationStyle.FRIENDLY: "uses warm, approachable language",
                CommunicationStyle.TECHNICAL: "employs precise technical terminology",
                CommunicationStyle.CASUAL: "speaks in a relaxed, conversational manner"
            }
            summary_parts.append(style_descriptions.get(
                personality.communication_style, 
                f"uses {personality.communication_style.value} communication"
            ))
            
            # Formality and expertise
            formality_desc = {
                FormalityLevel.FORMAL: "formal",
                FormalityLevel.CASUAL: "casual", 
                FormalityLevel.ADAPTIVE: "adaptable"
            }
            
            expertise_desc = {
                ExpertiseLevel.EXPERT: "expert-level",
                ExpertiseLevel.INTERMEDIATE: "intermediate-level",
                ExpertiseLevel.BEGINNER: "beginner-friendly"
            }
            
            summary_parts.append(f"with {formality_desc.get(personality.formality_level, 'standard')} tone")
            summary_parts.append(f"providing {expertise_desc.get(personality.expertise_level, 'general')} insights")
            
            # Response characteristics
            if personality.response_length == ResponseLength.CONCISE:
                summary_parts.append("in brief, focused responses")
            elif personality.response_length == ResponseLength.DETAILED:
                summary_parts.append("through comprehensive explanations")
            
            # Personality traits
            creativity_desc = "highly creative" if personality.creativity_level > 0.7 else \
                            "moderately creative" if personality.creativity_level > 0.4 else \
                            "structured and methodical"
            
            empathy_desc = "highly empathetic" if personality.empathy_level > 0.7 else \
                          "moderately empathetic" if personality.empathy_level > 0.4 else \
                          "direct and objective"
            
            summary_parts.append(f"This AI is {creativity_desc} and {empathy_desc}")
            
            # Custom traits
            if personality.custom_traits:
                trait_descriptions = []
                for trait, value in personality.custom_traits.items():
                    trait_descriptions.append(f"{trait.replace('_', ' ')}: {value}")
                
                if trait_descriptions:
                    summary_parts.append(f"with additional traits: {', '.join(trait_descriptions)}")
            
            return ". ".join(summary_parts) + "."
            
        except Exception as e:
            logger.error(f"Failed to generate personality summary: {e}")
            return "Personality configuration available but summary generation failed."
    
    # Private helper methods
    
    def _initialize_personality_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize personality templates for common roles"""
        return {
            "lawyer": {
                "communication_style": CommunicationStyle.PROFESSIONAL,
                "formality_level": FormalityLevel.FORMAL,
                "expertise_level": ExpertiseLevel.EXPERT,
                "creativity_adjustment": -0.2,
                "empathy_adjustment": 0.1,
                "traits": {"attention_to_detail": "high", "risk_tolerance": "low"}
            },
            "consultant": {
                "communication_style": CommunicationStyle.PROFESSIONAL,
                "formality_level": FormalityLevel.ADAPTIVE,
                "expertise_level": ExpertiseLevel.EXPERT,
                "creativity_adjustment": 0.1,
                "empathy_adjustment": 0.2,
                "traits": {"analytical_thinking": "high", "client_focus": "high"}
            },
            "engineer": {
                "communication_style": CommunicationStyle.TECHNICAL,
                "formality_level": FormalityLevel.CASUAL,
                "expertise_level": ExpertiseLevel.EXPERT,
                "creativity_adjustment": 0.2,
                "empathy_adjustment": -0.1,
                "traits": {"problem_solving": "high", "precision": "high"}
            },
            "support_agent": {
                "communication_style": CommunicationStyle.FRIENDLY,
                "formality_level": FormalityLevel.CASUAL,
                "expertise_level": ExpertiseLevel.INTERMEDIATE,
                "creativity_adjustment": 0.0,
                "empathy_adjustment": 0.3,
                "traits": {"patience": "high", "helpfulness": "high"}
            }
        }
    
    def _initialize_behavior_rules(self) -> Dict[BehaviorContext, Dict[str, Any]]:
        """Initialize behavior adaptation rules for different contexts"""
        return {
            BehaviorContext.PROFESSIONAL_MEETING: {
                "formality_boost": 0.3,
                "empathy_adjustment": 0.1,
                "creativity_limit": 0.6,
                "response_style": "structured"
            },
            BehaviorContext.CASUAL_CONVERSATION: {
                "formality_boost": -0.3,
                "empathy_adjustment": 0.2,
                "creativity_boost": 0.1,
                "response_style": "conversational"
            },
            BehaviorContext.TECHNICAL_DISCUSSION: {
                "precision_boost": 0.3,
                "creativity_limit": 0.4,
                "empathy_adjustment": -0.1,
                "response_style": "analytical"
            },
            BehaviorContext.CUSTOMER_SUPPORT: {
                "empathy_boost": 0.4,
                "patience_boost": 0.3,
                "formality_adjustment": 0.1,
                "response_style": "helpful"
            },
            BehaviorContext.CRISIS_MANAGEMENT: {
                "clarity_boost": 0.4,
                "empathy_boost": 0.2,
                "creativity_limit": 0.3,
                "response_style": "decisive"
            }
        }
    
    def _initialize_communication_patterns(self) -> Dict[str, List[str]]:
        """Initialize communication patterns for different styles"""
        return {
            "professional_formal": [
                "I would recommend...",
                "Based on my analysis...",
                "Please consider...",
                "It would be advisable to..."
            ],
            "friendly_casual": [
                "I'd love to help you with...",
                "Here's what I think...",
                "You might want to try...",
                "Let me share some ideas..."
            ],
            "technical_precise": [
                "The optimal approach is...",
                "According to the specifications...",
                "The implementation requires...",
                "Technical analysis indicates..."
            ]
        }
    
    def _apply_role_adjustments(
        self, 
        role: str, 
        industry: str, 
        creativity: float, 
        empathy: float
    ) -> Dict[str, float]:
        """Apply role-specific adjustments to personality traits"""
        
        adjusted_creativity = creativity
        adjusted_empathy = empathy
        
        # Role-based adjustments
        role_lower = role.lower()
        
        if "lawyer" in role_lower or "legal" in role_lower:
            adjusted_creativity = max(0.1, creativity - 0.2)  # Lawyers need structure
            adjusted_empathy = min(1.0, empathy + 0.1)       # But also client empathy
        
        elif "engineer" in role_lower or "developer" in role_lower:
            adjusted_creativity = min(1.0, creativity + 0.2)  # Engineers can be creative
            adjusted_empathy = max(0.2, empathy - 0.1)        # But focus on logic
        
        elif "support" in role_lower or "service" in role_lower:
            adjusted_empathy = min(1.0, empathy + 0.3)        # High empathy for support
        
        elif "analyst" in role_lower or "consultant" in role_lower:
            adjusted_creativity = min(1.0, creativity + 0.1)  # Moderate creativity boost
        
        # Industry-based adjustments
        industry_lower = industry.lower()
        
        if industry_lower in ["healthcare", "medical"]:
            adjusted_empathy = min(1.0, adjusted_empathy + 0.2)
        
        elif industry_lower in ["finance", "banking"]:
            adjusted_creativity = max(0.1, adjusted_creativity - 0.1)
        
        elif industry_lower in ["creative", "marketing", "advertising"]:
            adjusted_creativity = min(1.0, adjusted_creativity + 0.3)
        
        return {
            "creativity": adjusted_creativity,
            "empathy": adjusted_empathy
        }
    
    def _validate_personality_consistency(
        self,
        communication_style: CommunicationStyle,
        formality_level: FormalityLevel,
        expertise_level: ExpertiseLevel,
        creativity: float,
        empathy: float
    ) -> None:
        """Validate that personality traits are consistent"""
        
        # Check for major inconsistencies that should be errors
        if (communication_style == CommunicationStyle.TECHNICAL and 
            formality_level == FormalityLevel.CASUAL and 
            expertise_level == ExpertiseLevel.BEGINNER):
            raise InvalidConfigurationError(
                "Technical communication with casual formality and beginner expertise is inconsistent"
            )
        
        # Additional consistency checks can be added here
    
    def _get_role_specific_traits(self, role: str, industry: str) -> Dict[str, Any]:
        """Get role and industry specific traits"""
        
        traits = {}
        role_lower = role.lower()
        industry_lower = industry.lower()
        
        # Role-specific traits
        if "lawyer" in role_lower:
            traits.update({
                "attention_to_detail": "high",
                "risk_assessment": "conservative",
                "analytical_thinking": "high"
            })
        
        elif "engineer" in role_lower:
            traits.update({
                "problem_solving": "high",
                "systematic_approach": "high",
                "technical_precision": "high"
            })
        
        elif "consultant" in role_lower:
            traits.update({
                "strategic_thinking": "high",
                "client_focus": "high",
                "adaptability": "high"
            })
        
        # Industry-specific traits
        if industry_lower == "healthcare":
            traits.update({
                "patient_care": "high",
                "ethical_awareness": "high"
            })
        
        elif industry_lower == "finance":
            traits.update({
                "risk_management": "high",
                "regulatory_awareness": "high"
            })
        
        return traits
    
    def _get_context_adaptations(self, context: BehaviorContext) -> Dict[str, Any]:
        """Get context-specific behavior adaptations"""
        return self._behavior_rules.get(context, {})
    
    def _learn_from_interaction_history(
        self,
        history: List[Dict[str, Any]],
        current_traits: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Learn personality adaptations from interaction history"""
        
        adaptations = {}
        
        if not history:
            return adaptations
        
        # Analyze recent interactions for patterns
        recent_interactions = history[-5:]  # Last 5 interactions
        
        # Check for user feedback patterns
        positive_feedback_count = 0
        negative_feedback_count = 0
        
        for interaction in recent_interactions:
            feedback = interaction.get("user_feedback")
            if feedback:
                if feedback.get("satisfaction", 0) > 0.7:
                    positive_feedback_count += 1
                elif feedback.get("satisfaction", 0) < 0.3:
                    negative_feedback_count += 1
        
        # Adapt based on feedback patterns
        if negative_feedback_count > positive_feedback_count:
            # If getting negative feedback, slightly adjust empathy up
            adaptations["empathy_level"] = min(1.0, current_traits["empathy_level"] + 0.1)
        
        return adaptations
    
    def _generate_context_instructions(
        self,
        context: BehaviorContext,
        adapted_traits: Dict[str, Any]
    ) -> List[str]:
        """Generate context-specific behavioral instructions"""
        
        instructions = []
        
        if context == BehaviorContext.PROFESSIONAL_MEETING:
            instructions.extend([
                "Maintain professional decorum throughout the interaction",
                "Structure responses clearly with key points",
                "Avoid overly casual language or expressions"
            ])
        
        elif context == BehaviorContext.CUSTOMER_SUPPORT:
            instructions.extend([
                "Prioritize understanding the customer's concern",
                "Show empathy and patience in responses",
                "Provide clear, actionable solutions"
            ])
        
        elif context == BehaviorContext.TECHNICAL_DISCUSSION:
            instructions.extend([
                "Use precise technical terminology",
                "Provide detailed explanations with examples",
                "Focus on accuracy and completeness"
            ])
        
        elif context == BehaviorContext.CRISIS_MANAGEMENT:
            instructions.extend([
                "Remain calm and composed",
                "Provide clear, decisive guidance",
                "Focus on immediate actionable steps"
            ])
        
        return instructions
    
    def _get_optimal_model_for_task(
        self,
        task_type: str,
        personality: PersonalityConfig,
        default_model: str
    ) -> Optional[str]:
        """Get optimal model for specific task type and personality"""
        
        task_lower = task_type.lower()
        
        # Creative tasks benefit from higher creativity models
        if "creative" in task_lower or "brainstorm" in task_lower:
            if personality.creativity_level > 0.7:
                return "gpt-4"  # More creative model
        
        # Technical tasks benefit from precise models
        elif "technical" in task_lower or "code" in task_lower:
            return "gpt-4"  # More capable for technical tasks
        
        # Analysis tasks
        elif "analysis" in task_lower or "research" in task_lower:
            return "gpt-4"  # Better for complex analysis
        
        return None  # Use default model