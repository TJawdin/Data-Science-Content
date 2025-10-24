"""
Formatting utilities for Brazilian states and cities
Save as: utils/formatting.py
"""

import pandas as pd

# Brazilian State Names
STATE_NAMES = {
    'AC': 'Acre',
    'AL': 'Alagoas',
    'AP': 'Amapá',
    'AM': 'Amazonas',
    'BA': 'Bahia',
    'CE': 'Ceará',
    'DF': 'Distrito Federal',
    'ES': 'Espírito Santo',
    'GO': 'Goiás',
    'MA': 'Maranhão',
    'MT': 'Mato Grosso',
    'MS': 'Mato Grosso do Sul',
    'MG': 'Minas Gerais',
    'PA': 'Pará',
    'PB': 'Paraíba',
    'PR': 'Paraná',
    'PE': 'Pernambuco',
    'PI': 'Piauí',
    'RJ': 'Rio de Janeiro',
    'RN': 'Rio Grande do Norte',
    'RS': 'Rio Grande do Sul',
    'RO': 'Rondônia',
    'RR': 'Roraima',
    'SC': 'Santa Catarina',
    'SE': 'Sergipe',
    'SP': 'São Paulo',
    'TO': 'Tocantins'
}


def format_state_name(state_code):
    """
    Format state code to show both code and full name
    
    Args:
        state_code: Two-letter state code (e.g., 'SP', 'RJ')
        
    Returns:
        str: Formatted string like "SP - São Paulo"
    """
    if pd.isna(state_code) or state_code == '':
        return 'Unknown'
    
    state_code = str(state_code).upper().strip()
    full_name = STATE_NAMES.get(state_code, state_code)
    
    if full_name != state_code:
        return f"{state_code} - {full_name}"
    return state_code


def format_city_name(city_name):
    """
    Format city name to proper title case with accents
    
    Args:
        city_name: City name (may be lowercase)
        
    Returns:
        str: Properly formatted city name
    """
    if pd.isna(city_name) or city_name == '':
        return 'Unknown'
    
    city = str(city_name).strip()
    
    # Special cases for major Brazilian cities with proper accents
    special_cases = {
        'sao paulo': 'São Paulo',
        'rio de janeiro': 'Rio de Janeiro',
        'belo horizonte': 'Belo Horizonte',
        'porto alegre': 'Porto Alegre',
        'sao jose': 'São José',
        'sao bernardo do campo': 'São Bernardo do Campo',
        'sao goncalo': 'São Gonçalo',
        'sao caetano do sul': 'São Caetano do Sul',
        'campinas': 'Campinas',
        'curitiba': 'Curitiba',
        'brasilia': 'Brasília',
        'fortaleza': 'Fortaleza',
        'salvador': 'Salvador',
        'recife': 'Recife',
        'manaus': 'Manaus',
        'belem': 'Belém',
        'goiania': 'Goiânia',
        'guarulhos': 'Guarulhos',
        'maua': 'Mauá',
        'osasco': 'Osasco',
        'ribeirao preto': 'Ribeirão Preto',
        'santo andre': 'Santo André',
        'sorocaba': 'Sorocaba',
        'uberlandia': 'Uberlândia',
        'contagem': 'Contagem',
        'feira de santana': 'Feira de Santana',
        'joinville': 'Joinville',
        'londrina': 'Londrina',
        'niteroi': 'Niterói',
        'aparecida de goiania': 'Aparecida de Goiânia',
        'caxias do sul': 'Caxias do Sul',
        'juiz de fora': 'Juiz de Fora',
        'florianopolis': 'Florianópolis',
        'natal': 'Natal',
        'campo grande': 'Campo Grande',
        'teresina': 'Teresina',
        'sao luis': 'São Luís',
        'macapa': 'Macapá',
        'maceio': 'Maceió',
        'duque de caxias': 'Duque de Caxias',
        'nova iguacu': 'Nova Iguaçu',
        'betim': 'Betim',
        'caucaia': 'Caucaia'
    }
    
    city_lower = city.lower()
    if city_lower in special_cases:
        return special_cases[city_lower]
    
    # Default: Title case for other cities
    return city.title()


def get_state_abbreviation(state_name):
    """
    Get state abbreviation from full name
    
    Args:
        state_name: Full state name (e.g., 'São Paulo')
        
    Returns:
        str: Two-letter state code (e.g., 'SP')
    """
    if pd.isna(state_name) or state_name == '':
        return None
    
    # Reverse lookup
    state_name = str(state_name).strip()
    for code, name in STATE_NAMES.items():
        if name.lower() == state_name.lower():
            return code
    
    # If already a code, return it
    if len(state_name) == 2:
        return state_name.upper()
    
    return None
