"""
symbol_mapper.py
================
Dynamic stock symbol discovery and company name resolution.
Automatically discovers all available stocks from the dataset directory.

Features:
    - Dynamic stock discovery from CSV files
    - Company name to symbol resolution
    - Case-insensitive search
    - Fast lookup with caching
    - Backward compatible with existing code

Usage:
    from symbol_mapper import resolve_symbol, get_all_symbols, search_stocks
    
    # Resolve company name or symbol
    symbol = resolve_symbol("Tesla")  # Returns "TSLA"
    symbol = resolve_symbol("TSLA")   # Returns "TSLA"
    symbol = resolve_symbol("apple")  # Returns "AAPL"
    
    # Get all available symbols
    symbols = get_all_symbols()  # Returns ["AAPL", "TSLA", ...]
    
    # Search for stocks
    results = search_stocks("tech")  # Returns matching stocks
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from functools import lru_cache

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STOCKS_DIR = PROJECT_ROOT / "data" / "stock_market_dataset" / "stocks"
ETFS_DIR = PROJECT_ROOT / "data" / "stock_market_dataset" / "etfs"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Company Name Mapping
# ---------------------------------------------------------------------------

# Common company names for popular stocks
# This can be expanded as needed
SYMBOL_NAME_MAP = {
    # Tech Giants
    "AAPL": ["Apple", "Apple Inc", "Apple Computer"],
    "MSFT": ["Microsoft", "Microsoft Corporation"],
    "GOOGL": ["Google", "Alphabet", "Alphabet Inc"],
    "GOOG": ["Google", "Alphabet", "Alphabet Inc"],
    "AMZN": ["Amazon", "Amazon.com", "Amazon Inc"],
    "META": ["Meta", "Facebook", "Meta Platforms"],
    "FB": ["Facebook", "Meta", "Meta Platforms"],
    "NVDA": ["NVIDIA", "Nvidia", "Nvidia Corporation"],
    "TSLA": ["Tesla", "Tesla Inc", "Tesla Motors"],
    "NFLX": ["Netflix", "Netflix Inc"],
    
    # Other Major Tech
    "INTC": ["Intel", "Intel Corporation"],
    "AMD": ["AMD", "Advanced Micro Devices"],
    "ORCL": ["Oracle", "Oracle Corporation"],
    "CSCO": ["Cisco", "Cisco Systems"],
    "ADBE": ["Adobe", "Adobe Inc"],
    "CRM": ["Salesforce", "Salesforce.com"],
    "PYPL": ["PayPal", "PayPal Holdings"],
    "QCOM": ["Qualcomm", "Qualcomm Inc"],
    "AVGO": ["Broadcom", "Broadcom Inc"],
    "TXN": ["Texas Instruments", "TI"],
    
    # Retail & Consumer
    "WMT": ["Walmart", "Wal-Mart"],
    "COST": ["Costco", "Costco Wholesale"],
    "HD": ["Home Depot", "The Home Depot"],
    "NKE": ["Nike", "Nike Inc"],
    "SBUX": ["Starbucks", "Starbucks Corporation"],
    "MCD": ["McDonald's", "McDonalds"],
    "DIS": ["Disney", "Walt Disney", "The Walt Disney Company"],
    
    # Finance
    "JPM": ["JPMorgan", "JP Morgan", "JPMorgan Chase"],
    "BAC": ["Bank of America", "BofA"],
    "WFC": ["Wells Fargo", "Wells Fargo & Company"],
    "GS": ["Goldman Sachs", "Goldman Sachs Group"],
    "MS": ["Morgan Stanley"],
    "V": ["Visa", "Visa Inc"],
    "MA": ["Mastercard", "MasterCard"],
    
    # Healthcare
    "JNJ": ["Johnson & Johnson", "Johnson and Johnson", "J&J"],
    "UNH": ["UnitedHealth", "United Health Group"],
    "PFE": ["Pfizer", "Pfizer Inc"],
    "ABBV": ["AbbVie", "AbbVie Inc"],
    "TMO": ["Thermo Fisher", "Thermo Fisher Scientific"],
    
    # Energy
    "XOM": ["Exxon", "ExxonMobil", "Exxon Mobil"],
    "CVX": ["Chevron", "Chevron Corporation"],
    
    # Telecom
    "T": ["AT&T", "ATT"],
    "VZ": ["Verizon", "Verizon Communications"],
    
    # Automotive
    "F": ["Ford", "Ford Motor", "Ford Motor Company"],
    "GM": ["General Motors", "GM"],
    
    # Food & Beverage
    "KO": ["Coca-Cola", "Coca Cola", "Coke"],
    "PEP": ["Pepsi", "PepsiCo"],
}

# Build reverse lookup: name -> symbol
_NAME_TO_SYMBOL_CACHE: Optional[Dict[str, str]] = None


def _build_name_to_symbol_map() -> Dict[str, str]:
    """
    Build reverse lookup map from company names to symbols.
    All names are normalized to lowercase for case-insensitive matching.
    
    Returns:
        Dictionary mapping normalized company name to symbol
    """
    name_map = {}
    
    for symbol, names in SYMBOL_NAME_MAP.items():
        for name in names:
            normalized = name.lower().strip()
            # Store the first occurrence (prefer shorter symbols like GOOGL over GOOG)
            if normalized not in name_map:
                name_map[normalized] = symbol
    
    return name_map


def get_name_to_symbol_map() -> Dict[str, str]:
    """
    Get cached name-to-symbol mapping.
    
    Returns:
        Dictionary mapping normalized company name to symbol
    """
    global _NAME_TO_SYMBOL_CACHE
    
    if _NAME_TO_SYMBOL_CACHE is None:
        _NAME_TO_SYMBOL_CACHE = _build_name_to_symbol_map()
    
    return _NAME_TO_SYMBOL_CACHE


# ---------------------------------------------------------------------------
# Dynamic Stock Discovery
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def discover_stocks_from_directory(directory: Path) -> List[str]:
    """
    Discover all stock symbols from CSV files in a directory.
    Results are cached for performance.
    
    Args:
        directory: Path to directory containing stock CSV files
        
    Returns:
        Sorted list of stock symbols (without .csv extension)
    """
    if not directory.exists():
        logger.warning(f"Stock directory not found: {directory}")
        return []
    
    try:
        symbols = sorted([
            csv_file.stem.upper()
            for csv_file in directory.glob("*.csv")
            if csv_file.is_file()
        ])
        
        logger.info(f"Discovered {len(symbols)} stocks from {directory.name}")
        return symbols
        
    except Exception as e:
        logger.error(f"Failed to discover stocks from {directory}: {e}")
        return []


def get_all_symbols(include_etfs: bool = False) -> List[str]:
    """
    Get all available stock symbols from the dataset.
    
    Args:
        include_etfs: Whether to include ETFs in addition to stocks
        
    Returns:
        Sorted list of all available stock symbols
    """
    symbols = discover_stocks_from_directory(STOCKS_DIR)
    
    if include_etfs:
        etf_symbols = discover_stocks_from_directory(ETFS_DIR)
        symbols = sorted(set(symbols + etf_symbols))
    
    return symbols


def get_stock_count() -> int:
    """
    Get total count of available stocks.
    
    Returns:
        Number of available stock symbols
    """
    return len(get_all_symbols())


# ---------------------------------------------------------------------------
# Symbol Resolution
# ---------------------------------------------------------------------------

def resolve_symbol(user_input: str) -> str:
    """
    Resolve user input to a valid stock symbol.
    
    Supports:
        - Direct symbol lookup (case-insensitive): "AAPL", "aapl", "Aapl"
        - Company name lookup: "Apple", "apple", "APPLE"
        - Partial matching for company names
    
    Args:
        user_input: User-provided stock symbol or company name
        
    Returns:
        Normalized uppercase stock symbol
        
    Raises:
        ValueError: If symbol cannot be resolved or is not available
    """
    if not user_input or not user_input.strip():
        raise ValueError("Stock symbol or company name cannot be empty")
    
    # Normalize input
    normalized_input = user_input.strip().upper()
    
    # Get all available symbols
    available_symbols = get_all_symbols()
    
    # Step 1: Direct symbol match (case-insensitive)
    if normalized_input in available_symbols:
        logger.debug(f"Resolved '{user_input}' -> {normalized_input} (direct match)")
        return normalized_input
    
    # Step 2: Company name lookup
    name_map = get_name_to_symbol_map()
    normalized_name = user_input.strip().lower()
    
    if normalized_name in name_map:
        symbol = name_map[normalized_name]
        
        # Verify symbol is available in dataset
        if symbol in available_symbols:
            logger.debug(f"Resolved '{user_input}' -> {symbol} (company name match)")
            return symbol
        else:
            logger.warning(f"Symbol {symbol} found in name map but not in dataset")
    
    # Step 3: Partial company name matching
    for name, symbol in name_map.items():
        if normalized_name in name or name in normalized_name:
            if symbol in available_symbols:
                logger.debug(f"Resolved '{user_input}' -> {symbol} (partial name match)")
                return symbol
    
    # Not found
    raise ValueError(
        f"Stock '{user_input}' not found.\n"
        f"Available stocks: {len(available_symbols)} symbols in dataset.\n"
        f"Use get_all_symbols() to see all available stocks."
    )


def is_symbol_available(symbol: str) -> bool:
    """
    Check if a stock symbol is available in the dataset.
    
    Args:
        symbol: Stock ticker symbol
        
    Returns:
        True if symbol is available, False otherwise
    """
    try:
        normalized = symbol.strip().upper()
        return normalized in get_all_symbols()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Search Functionality
# ---------------------------------------------------------------------------

def search_stocks(query: str, limit: int = 20) -> List[Tuple[str, Optional[str]]]:
    """
    Search for stocks by symbol or company name.
    
    Args:
        query: Search query (symbol or company name)
        limit: Maximum number of results to return
        
    Returns:
        List of tuples (symbol, company_name) matching the query
        Company name is None if not in the mapping
    """
    if not query or not query.strip():
        return []
    
    query_lower = query.strip().lower()
    results = []
    available_symbols = get_all_symbols()
    
    # Build symbol -> name mapping for display
    symbol_to_names = {}
    for symbol, names in SYMBOL_NAME_MAP.items():
        if names:
            symbol_to_names[symbol] = names[0]  # Use first (primary) name
    
    # Search by symbol
    for symbol in available_symbols:
        if query_lower in symbol.lower():
            company_name = symbol_to_names.get(symbol)
            results.append((symbol, company_name))
            
            if len(results) >= limit:
                return results
    
    # Search by company name
    name_map = get_name_to_symbol_map()
    for name, symbol in name_map.items():
        if query_lower in name and symbol in available_symbols:
            if symbol not in [r[0] for r in results]:  # Avoid duplicates
                company_name = symbol_to_names.get(symbol)
                results.append((symbol, company_name))
                
                if len(results) >= limit:
                    return results
    
    return results


def get_company_name(symbol: str) -> Optional[str]:
    """
    Get company name for a given symbol.
    
    Args:
        symbol: Stock ticker symbol
        
    Returns:
        Company name if available, None otherwise
    """
    normalized = symbol.strip().upper()
    names = SYMBOL_NAME_MAP.get(normalized)
    return names[0] if names else None


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def validate_and_resolve(user_input: str) -> str:
    """
    Validate and resolve user input to a stock symbol.
    This is the main entry point for user input processing.
    
    Args:
        user_input: User-provided stock symbol or company name
        
    Returns:
        Validated and normalized stock symbol
        
    Raises:
        ValueError: If input is invalid or symbol not found
    """
    return resolve_symbol(user_input)


def get_supported_stocks() -> List[str]:
    """
    Get list of all supported stock symbols.
    Alias for get_all_symbols() for backward compatibility.
    
    Returns:
        List of all available stock symbols
    """
    return get_all_symbols()


# ---------------------------------------------------------------------------
# CLI Testing
# ---------------------------------------------------------------------------

def main():
    """CLI tool for testing symbol resolution."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python symbol_mapper.py <symbol_or_name>")
        print("\nExamples:")
        print("  python symbol_mapper.py AAPL")
        print("  python symbol_mapper.py apple")
        print("  python symbol_mapper.py Tesla")
        print("  python symbol_mapper.py --list")
        print("  python symbol_mapper.py --search tech")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "--list":
        symbols = get_all_symbols()
        print(f"\nTotal available stocks: {len(symbols)}")
        print("\nFirst 50 symbols:")
        for symbol in symbols[:50]:
            name = get_company_name(symbol)
            if name:
                print(f"  {symbol:6s} - {name}")
            else:
                print(f"  {symbol}")
        if len(symbols) > 50:
            print(f"\n... and {len(symbols) - 50} more")
    
    elif command == "--search":
        if len(sys.argv) < 3:
            print("Usage: python symbol_mapper.py --search <query>")
            sys.exit(1)
        
        query = sys.argv[2]
        results = search_stocks(query, limit=20)
        
        print(f"\nSearch results for '{query}': {len(results)} matches")
        for symbol, name in results:
            if name:
                print(f"  {symbol:6s} - {name}")
            else:
                print(f"  {symbol}")
    
    elif command == "--count":
        count = get_stock_count()
        print(f"\nTotal available stocks: {count}")
    
    else:
        try:
            symbol = resolve_symbol(command)
            name = get_company_name(symbol)
            print(f"\nResolved: '{command}' -> {symbol}")
            if name:
                print(f"Company: {name}")
        except ValueError as e:
            print(f"\nError: {e}")
            sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
