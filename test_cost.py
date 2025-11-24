import boto3
import json
from datetime import datetime, timedelta

# AWS SageMaker pricing (US East - N. Virginia) as of 2024
# Update these values by checking: https://aws.amazon.com/sagemaker/pricing/
INSTANCE_PRICING = {
    'ml.t3.medium': 0.0416,
    'ml.t3.large': 0.0832,
    'ml.g4dn.xlarge': 0.736,
    'ml.g5.xlarge': 1.408,
    'ml.g5.2xlarge': 2.03,
    'ml.g5.4xlarge': 4.06,
    'ml.p3.2xlarge': 3.825,
}

# Your models configuration
MODELS = {
    'Scratch Models': [
        {'name': 'UpConv', 'params': 0.000364, 'instance': 'ml.t3.medium'},
        {'name': 'LGrad', 'params': 23.510081, 'instance': 'ml.g4dn.xlarge'},
        {'name': 'NPR', 'params': 1.437761, 'instance': 'ml.t3.large'},
        {'name': 'FrqNet', 'params': 1.850433, 'instance': 'ml.t3.large'},
        {'name': 'AIDE', 'params': 54.432466, 'instance': 'ml.g5.xlarge'},
        {'name': 'DMD', 'params': 23.510081, 'instance': 'ml.g4dn.xlarge'},
        {'name': 'AlgnF', 'params': 23.510081, 'instance': 'ml.g4dn.xlarge'},
        {'name': 'StayP', 'params': 23.510081, 'instance': 'ml.g4dn.xlarge'},
    ],
    'Frozen Models': [
        {'name': 'UniD', 'params': 427.617282, 'instance': 'ml.g5.2xlarge'},
        {'name': 'RCLip', 'params': 0.001025, 'instance': 'ml.t3.medium'},
        {'name': 'RINE', 'params': 6.323201, 'instance': 'ml.g4dn.xlarge'},
        {'name': 'C2PClip', 'params': 0.000769, 'instance': 'ml.t3.medium'},
    ],
    'Fine-Tuned Models': [
        {'name': 'CNND', 'params': 23.510081, 'instance': 'ml.g4dn.xlarge'},
        {'name': 'FatF', 'params': 492.591044, 'instance': 'ml.g5.4xlarge'},
        {'name': 'EFFT', 'params': 492.591044, 'instance': 'ml.g5.4xlarge'},
        {'name': 'ForL', 'params': 14.189569, 'instance': 'ml.g4dn.xlarge'},
    ]
}

def calculate_costs(hours_per_day=24, days_per_month=30):
    """Calculate costs for different usage patterns"""
    
    total_hourly = 0
    total_daily = 0
    total_monthly = 0
    
    print(f"\n{'='*80}")
    print(f"AWS SAGEMAKER DEPLOYMENT COST ESTIMATION")
    print(f"Usage Pattern: {hours_per_day} hours/day, {days_per_month} days/month")
    print(f"{'='*80}\n")
    
    for category, models in MODELS.items():
        print(f"\n{category}:")
        print(f"{'-'*80}")
        category_cost = 0
        
        for model in models:
            hourly_rate = INSTANCE_PRICING[model['instance']]
            daily_cost = hourly_rate * hours_per_day
            monthly_cost = daily_cost * days_per_month
            
            print(f"  {model['name']:12} | {model['instance']:18} | "
                  f"${hourly_rate:6.3f}/hr | ${monthly_cost:8.2f}/month")
            
            category_cost += monthly_cost
            total_hourly += hourly_rate
            total_daily += daily_cost
            total_monthly += monthly_cost
        
        print(f"  {'-'*78}")
        print(f"  Category Total: ${category_cost:,.2f}/month")
    
    print(f"\n{'='*80}")
    print(f"TOTAL COST:")
    print(f"  Hourly:  ${total_hourly:,.2f}")
    print(f"  Daily:   ${total_daily:,.2f}")
    print(f"  Monthly: ${total_monthly:,.2f}")
    print(f"  Yearly:  ${total_monthly * 12:,.2f}")
    print(f"{'='*80}\n")
    
    return total_monthly

def compare_scenarios():
    """Compare different deployment scenarios"""
    
    print("\n" + "="*80)
    print("COST COMPARISON: DIFFERENT SCENARIOS")
    print("="*80)
    
    scenarios = [
        ("24/7 Production", 24, 30),
        ("Business Hours (8hrs/day)", 8, 22),
        ("Development (4hrs/day)", 4, 22),
        ("Weekend Only", 8, 8),
    ]
    
    for name, hours, days in scenarios:
        cost = calculate_costs(hours, days)
        print(f"\n{name}: ${cost:,.2f}/month\n")

def fetch_live_pricing():
    """Fetch live pricing from AWS (requires AWS credentials)"""
    try:
        pricing_client = boto3.client('pricing', region_name='us-east-2')
        
        print("\nFetching live pricing from AWS...")
        
        for instance_type in INSTANCE_PRICING.keys():
            response = pricing_client.get_products(
                ServiceCode='AmazonSageMaker',
                Filters=[
                    {'Type': 'TERM_MATCH', 'Field': 'instanceType', 'Value': instance_type},
                    {'Type': 'TERM_MATCH', 'Field': 'location', 'Value': 'US East (N. Virginia)'},
                ]
            )
            
            if response['PriceList']:
                price_data = json.loads(response['PriceList'][0])
                on_demand = price_data['terms']['OnDemand']
                price_dimensions = list(on_demand.values())[0]['priceDimensions']
                price_per_hour = float(list(price_dimensions.values())[0]['pricePerUnit']['USD'])
                print(f"{instance_type}: ${price_per_hour:.3f}/hour")
    
    except Exception as e:
        print(f"Error fetching live pricing: {e}")
        print("Using cached pricing data instead.")

if __name__ == "__main__":
    # Calculate default scenario (24/7)
    calculate_costs(hours_per_day=24, days_per_month=30)
    
    # Compare different scenarios
    compare_scenarios()
    
    # Uncomment to fetch live pricing (requires AWS credentials)
    # fetch_live_pricing()