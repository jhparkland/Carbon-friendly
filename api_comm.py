import requests
import schedule
import time

def call_api(country_code):
    url = f'https://app.electricitymaps.com/zone/{country_code}'
    response = requests.get(url)

    if response.status_code == 200:
        print(f"API call successful for {country_code} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        # You can process the API response here if needed
    else:
        print(f"API call failed for {country_code} with status code {response.status_code}")

# List of country codes
countries = ['DE', 'FR', 'US', 'KR', 'JP']

# Schedule API calls for each country every hour
for country_code in countries:
    schedule.every().hour.do(call_api, country_code)

# Run the scheduler in an infinite loop
while True:
    schedule.run_pending()
    time.sleep(1)