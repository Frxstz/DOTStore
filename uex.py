import requests
import json
import csv
from typing import Optional, Dict, List, Any

class UEXMarketplace:
    """Client for fetching marketplace data from UEX Corp API"""

    BASE_URL = "https://api.uexcorp.uk/2.0/marketplace_averages_all"

    def __init__(self):
        self.data = None

    def fetch_data(self) -> Dict[str, Any]:
        """Fetch marketplace data from the API"""
        try:
            response = requests.get(self.BASE_URL)
            response.raise_for_status()
            self.data = response.json()
            return self.data
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            return None

    def get_all_items(self) -> List[Dict[str, Any]]:
        """Get all items from the marketplace data"""
        if not self.data:
            self.fetch_data()

        if self.data and "data" in self.data:
            return self.data["data"]
        return []

    def get_item_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Find an item by its name"""
        items = self.get_all_items()
        for item in items:
            if item.get("item_name", "").lower() == name.lower():
                return item
        return None

    def get_items_by_category(self, category_id: int) -> List[Dict[str, Any]]:
        """Get all items in a specific category"""
        items = self.get_all_items()
        return [item for item in items if item.get("id_category") == category_id]

    def get_best_selling_prices(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get items with the highest selling prices"""
        items = self.get_all_items()
        sorted_items = sorted(
            items,
            key=lambda x: x.get("price_sell", 0) or 0,
            reverse=True
        )
        return sorted_items[:limit]

    def save_to_csv(self, filename: str = "marketplace_data.csv") -> bool:
        """Save all marketplace items to a CSV file"""
        items = self.get_all_items()

        if not items:
            print("No data to save")
            return False

        try:
            # Get all unique field names from the items
            fieldnames = set()
            for item in items:
                fieldnames.update(item.keys())
            fieldnames = sorted(fieldnames)

            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(items)

            print(f"Successfully saved {len(items)} items to {filename}")
            return True
        except Exception as e:
            print(f"Error saving to CSV: {e}")
            return False

    def print_item_summary(self, item: Dict[str, Any]) -> None:
        """Print a formatted summary of an item"""
        print(f"\nItem: {item.get('item_name', 'Unknown')}")
        print(f"  ID: {item.get('id_item', 'N/A')}")
        print(f"  Category ID: {item.get('id_category', 'N/A')}")
        print(f"  Buy Price: {item.get('price_buy', 'N/A')}")
        print(f"  Sell Price: {item.get('price_sell', 'N/A')}")
        print(f"  Date Modified: {item.get('date_modified', 'N/A')}")


def main():
    """Example usage of the UEXMarketplace client"""
    marketplace = UEXMarketplace()

    # Fetch data
    print("Fetching marketplace data...")
    data = marketplace.fetch_data()

    if data:
        print(f"Status: {data.get('status')}")
        print(f"Total items: {len(marketplace.get_all_items())}")

        # Save to CSV
        marketplace.save_to_csv("marketplace_data.csv")

        # Get top 5 most expensive items
        print("\nTop 5 most expensive items:")
        top_items = marketplace.get_best_selling_prices(limit=5)
        for item in top_items:
            marketplace.print_item_summary(item)


if __name__ == "__main__":
    main()
