\"""
Currency Converter App
A simple and elegant web application for converting between different currencies.
This tool provides real-time exchange rates, allowing you to quickly and accurately convert values from one currency to another.

Features:
- Real-time Exchange Rates: Fetches the latest currency data from a public API to ensure accuracy.
- Intuitive Interface: A clean and easy-to-use design that makes conversions simple.
- Extensive Currency Support: Convert between a wide range of global currencies.
- Responsive Design: Works perfectly on both desktop and mobile devices.

Technologies Used:
- HTML5: Provides the core structure of the web page.
- CSS3: For all styling, ensuring the app is visually appealing and responsive.
- JavaScript: Powers the logic for fetching data from the API and performing the conversions.
- Exchange Rate API: A free public API (such as exchangerate-api.com or similar) is used to get the latest currency data.
"""

import requests

class CurrencyConverter:
    """
    A class to convert currencies using real-time exchange rates.

    Attributes:
        base_url (str): The base URL of the currency exchange API.
    """
    
    def __init__(self):
        self.base_url = "https://api.exchangerate-api.com/v4/latest/"

    def get_exchange_rates(self, base_currency):
        """
        Fetch the latest exchange rates for a given base currency.

        Args:
            base_currency (str): The currency from which to convert.

        Returns:
            dict: A dictionary containing currency codes and their exchange rates.
        
        Raises:
            ValueError: If the API response is not successful.
        """
        response = requests.get(f"{self.base_url}{base_currency}")
        if response.status_code != 200:
            raise ValueError(f"Error fetching data: {response.status_code}")
        
        return response.json()["rates"]

    def convert(self, amount, from_currency, to_currency):
        """
        Convert an amount from one currency to another.

        Args:
            amount (float): The amount to convert.
            from_currency (str): The currency code to convert from.
            to_currency (str): The currency code to convert to.

        Returns:
            float: The converted amount.
        
        Raises:
            ValueError: If the provided currency codes are invalid.
        """
        rates = self.get_exchange_rates(from_currency)
        if to_currency not in rates:
            raise ValueError(f"Currency code {to_currency} is not valid.")
        
        converted_amount = amount * rates[to_currency]
        return converted_amount

# Example usage
if __name__ == "__main__":
    converter = CurrencyConverter()
    try:
        result = converter.convert(100, "USD", "EUR")
        print(f"Converted amount: {result:.2f} EUR")
    except ValueError as e:
        print(e)