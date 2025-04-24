import streamlit as st
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline

class CampaignRevenuePredictorApp:
    def __init__(self):
        self.title = " Campaign Payout Predictor"
        st.set_page_config(page_title=self.title, layout="centered")
        self.run()

    def run(self):
        st.title(self.title)
        st.markdown("Fill in the details below to predict Payout:")

        with st.form("prediction_form"):
            Amount = st.number_input("Amount", min_value=0.0, format="%.2f")
            Campaign_Name = st.text_input("Campaign Name")
            Sub_Id = st.text_input("Sub ID")
            Partner = st.text_input("Partner")
            submitted = st.form_submit_button("Predict")

        if submitted:
            self.predict(Amount, Campaign_Name, Sub_Id, Partner)

    def predict(self, Amount, Campaign_Name, Sub_Id, Partner):
        try:
            # Prepare data using your CustomData class
            data = CustomData(
                Amount=Amount,
                Campaign_Name=Campaign_Name,
                Sub_Id=Sub_Id,
                Partner=Partner
            )

            final_input_df = data.get_data_as_dataframe()

            # Run prediction using your PredictPipeline class
            prediction_pipeline = PredictPipeline()
            pred = prediction_pipeline.predict(final_input_df)

            # Show result
            result = round(pred[0], 2)
            st.success(f" Predicted Payout: ₹ {result}")

        except Exception as e:
            st.error(f" Error during prediction: {e}")

# Run the app
if __name__ == "__main__":
    CampaignRevenuePredictorApp()
