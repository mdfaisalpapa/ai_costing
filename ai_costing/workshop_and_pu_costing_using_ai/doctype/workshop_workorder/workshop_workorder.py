import frappe
from frappe.model.document import Document
import os

class WorkshopWorkorder(Document):
    def before_save(self):
        # Ensure required inputs exist before attempting a prediction
        if not self.decoded_workorder or not self.shop or not self.direct_hours:
            return
            
        try:
            import pandas as pd
            from catboost import CatBoostRegressor
            
            # Resolve the absolute path to the .cbm model file we copied earlier
            model_path = os.path.join(os.path.dirname(__file__), 'real_wgr_costing_model.cbm')
            
            # Load the model
            model = CatBoostRegressor()
            model.load_model(model_path)
            
            # Format the data exactly as the model expects
            input_data = pd.DataFrame([{
                'Decoded Workorder': self.decoded_workorder,
                'Shop': self.shop,
                'Direct Hours': self.direct_hours
            }])
            
            # Execute prediction
            prediction = model.predict(input_data)[0]
            self.ai_predicted_cost = prediction
            
            # Calculate variance if actual cost (Grand Total) is known
            if self.grand_total:
                variance = ((self.grand_total - self.ai_predicted_cost) / self.ai_predicted_cost) * 100
                self.variance_pct = variance
                
        except Exception as e:
            frappe.log_error(f"AI Costing Prediction Error: {str(e)}", "AI Prediction Failure")
            frappe.msgprint("AI Prediction failed. Check Error Logs for details.")