# currency.py

import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CurrencyConverter:
    def __init__(self):
        self.currencies = ["USD", "EUR", "GBP"]  # Existing currencies
        self.exchange_rates = {}

    def add_currency(self, currency_code):
        """Adds a new currency to the currency list."""
        if currency_code not in self.currencies:
            self.currencies.append(currency_code)
            logger.info(f"Currency {currency_code} added successfully.")
        else:
            logger.warning(f"Currency {currency_code} already exists.")

    def fetch_exchange_rates(self):
        """Fetches the exchange rates for all currencies including JPY."""
        try:
            response = requests.get("https://api.exchangerate-api.com/v4/latest/USD")
            data = response.json()
            self.exchange_rates = data['rates']
            logger.info("Exchange rates fetched successfully.")
        except Exception as e:
            logger.error(f"Error fetching exchange rates: {e}")

    def convert(self, amount, from_currency, to_currency):
        """Converts amount from one currency to another."""
        if from_currency not in self.currencies or to_currency not in self.currencies:
            logger.error("Currency not supported.")
            return None
        
        if from_currency == to_currency:
            return amount
        
        from_rate = self.exchange_rates.get(from_currency)
        to_rate = self.exchange_rates.get(to_currency)
        
        if from_rate is None or to_rate is None:
            logger.error("Exchange rate not found.")
            return None
        
        converted_amount = (amount / from_rate) * to_rate
        logger.info(f"Converted {amount} {from_currency} to {converted_amount} {to_currency}.")
        return converted_amount

# Usage
if __name__ == "__main__":
    converter = CurrencyConverter()
    converter.add_currency("JPY")
    converter.fetch_exchange_rates()
    print(converter.convert(100, "JPY", "USD"))
