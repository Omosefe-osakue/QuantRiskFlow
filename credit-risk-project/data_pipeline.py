import pandas as pd
import numpy as np
from faker import Faker
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
class Pipeline:
    def create_sample_data(self):
        fake = Faker()

        # Parameters
        num_customers = 1000
        num_features = 12

        # Generate sample data
        customer_id = [fake.unique.uuid4() for _ in range(num_customers)]
        full_name = [fake.name() for _ in range(num_customers)]
        email = [fake.email() for _ in range(num_customers)]
        phone_number = [fake.phone_number() for _ in range(num_customers)]
        address = [fake.address() for _ in range(num_customers)]
        credit_score = [fake.random_int(min=300, max=850) for _ in range(num_customers)]
        income = [fake.random_int(min=20000, max=900000) for _ in range(num_customers)]
        account_balance = [fake.random_int(min=0, max=1000000) for _ in range(num_customers)]
        loan_amount = [fake.random_int(min=1000, max=10000) for _ in range(num_customers)]
        loan_term = [fake.random_int(min=1, max=30) for _ in range(num_customers)]
        loan_purpose = [fake.random_element(elements=("Home Purchase", "Business Loan", "Auto Loan", "Student Loan")) for _ in range(num_customers)]
        loan_status = [fake.random_element(elements=("Approved", "Rejected", "Pending")) for _ in range(num_customers)]
        credit_utilization = [fake.random_int(min=0, max=100) for _ in range(num_customers)]
        
        data = {
            'CustomerID': customer_id,
            'FullName':full_name,
            'Email':email,
            'PhoneNumber':phone_number,
            'Address':address,
            'CreditScore':credit_score,
            'Income':income,
            'AccountBalance':account_balance,
            'LoanAmount':loan_amount,
            'LoanTerm':loan_term,
            'LoanPurpose':loan_purpose,
            'LoanStatus':loan_status,
            'CreditUtilization':credit_utilization
        }

        df = pd.DataFrame(data)
        df.to_csv("fake_sample_data.csv", index=False)
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Engineering features...")
        # loan_to_income_ratio: loan amount / income
        df['loan_to_income_ratio'] = df['LoanAmount'] / df['Income']
        # high_utilization: if credit utilization is greater than 80%
        df['high_utilization'] = df['CreditUtilization'] > 80
    
if __name__ == "__main__":
    pipeline = Pipeline()
    df = pipeline.create_sample_data()
    print("Sample Data Preview:")
    print(df.head())    