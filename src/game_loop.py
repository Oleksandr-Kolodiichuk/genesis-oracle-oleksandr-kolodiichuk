import os
import json
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from sandbox_env import ThermalDampener

# 1. Define the strict structured contract using Pydantic
class ControlDecision(BaseModel):
    system_state: str = Field(description="Must be 'FREEZING', 'BOILING', or 'PERFECT'")
    adjustment_action: str = Field(description="Must be 'INCREASE', 'DECREASE', or 'HOLD'")
    delta_value: float = Field(description="The exact numerical change to apply to Kappa")
    confidence_score: float

def main():
    # Instantiate the client
    client = genai.Client()
    
    # Initialize the sandbox with a highly volatile starting state (Freezing)
    env = ThermalDampener(initial_kappa=1.2)
    
    print("Initiating Parameter Hide-and-Seek Control Loop...\n")

    # 2. The Game Loop: 5 consecutive turns
    for turn in range(1, 6):
        current_temp = env.get_temperature()
        print(f"--- Turn {turn} ---")
        print(f"Current Kappa: {env.kappa:.2f} | Current Temperature: {current_temp:.2f}°C")

        # Formulate the prompt with the current telemetry
        prompt = (
            f"You are the control unit for a thermal dampener. The target optimal temperature is 50.0°C. "
            f"The current temperature is {current_temp:.2f}°C. "
            f"If temperature < 45°C, state is FREEZING. If > 55°C, state is BOILING. Otherwise, it is PERFECT. "
            f"Assume the physical equation is roughly: Temperature = Kappa * 10. "
            f"Provide the exact JSON adjustment to reach the PERFECT state."
        )

        try:
            # Enforce the ControlDecision schema on the Gemini model
            response = client.models.generate_content(
                model='gemini-2.0-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ControlDecision,
                    temperature=0.1, # Low temperature for analytical consistency
                ),
            )
            
            # Parse the payload programmatically
            payload = json.loads(response.text)
            decision = ControlDecision(**payload)

            print(f"Model JSON Payload:\n{json.dumps(payload, indent=2)}")
            print(f"Action: {decision.adjustment_action} Kappa by {decision.delta_value}")

            # Actively modify Kappa
            env.update_kappa(decision.delta_value)

        except Exception as e:
            print(f"API Error or JSON Parsing Failure: {e}")
            break

        print("-" * 40)

    print(f"Final System State -> Kappa: {env.kappa:.2f} | Temperature: {env.get_temperature():.2f}°C")

if __name__ == "__main__":
    main()