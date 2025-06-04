#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 07:18:12 2025

@author: rameshbk
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Domain Knowledge Module for Emission Regulation Assistant
Contains specialized automotive emission knowledge to enhance RAG capabilities
"""

# Vehicle categorization by type and fuel
VEHICLE_CATEGORIES = {
    "L1": {"description": "Two-wheeled vehicle with engine capacity ≤ 50 cm³ and max speed ≤ 50 km/h"},
    "L2": {"description": "Three-wheeled vehicle with engine capacity ≤ 50 cm³ and max speed ≤ 50 km/h"},
    "L3": {"description": "Two-wheeled vehicle with engine capacity > 50 cm³ or max speed > 50 km/h"},
    "L4": {"description": "Vehicle with three wheels asymmetrically arranged with engine capacity > 50 cm³ or max speed > 50 km/h"},
    "L5": {"description": "Vehicle with three wheels symmetrically arranged with engine capacity > 50 cm³ or max speed > 50 km/h"},
    "M1": {"description": "Passenger vehicles with seating capacity ≤ 9 persons"},
    "M2": {"description": "Passenger vehicles with seating capacity > 9 persons and GVW ≤ 5 tonnes"},
    "M3": {"description": "Passenger vehicles with seating capacity > 9 persons and GVW > 5 tonnes"},
    "N1": {"description": "Goods vehicles with GVW ≤ 3.5 tonnes"},
    "N2": {"description": "Goods vehicles with GVW > 3.5 tonnes but ≤ 12 tonnes"},
    "N3": {"description": "Goods vehicles with GVW > 12 tonnes"},
}

# Fuel types and their characteristics
FUEL_TYPES = {
    "Petrol": {
        "emission_concerns": ["CO", "HC", "NOx", "Evaporative emissions", "Particulate matter (for DI engines)"],
        "control_systems": ["TWC", "EGR", "EVP", "OBD", "PCV"]
    },
    "Diesel": {
        "emission_concerns": ["NOx", "PM", "PN", "CO", "HC"],
        "control_systems": ["DOC", "DPF", "SCR", "EGR", "OBD"]
    },
    "CNG": {
        "emission_concerns": ["CO", "HC", "NOx", "Methane"],
        "control_systems": ["TWC", "EGR", "OBD"]
    },
    "LPG": {
        "emission_concerns": ["CO", "HC", "NOx"],
        "control_systems": ["TWC", "EGR", "OBD"]
    },
    "Electric": {
        "emission_concerns": [],
        "control_systems": []
    },
    "Hybrid": {
        "emission_concerns": ["CO", "HC", "NOx", "Evaporative emissions"],
        "control_systems": ["TWC", "EGR", "EVP", "OBD"]
    }
}

# Emission control systems and their applicability
EMISSION_CONTROL_SYSTEMS = {
    "TWC": {
        "full_name": "Three-Way Catalyst",
        "description": "Catalytic converter that reduces NOx, CO, and HC emissions",
        "applicable_fuels": ["Petrol", "CNG", "LPG"],
        "not_applicable_fuels": ["Diesel", "Electric"],
        "required_for": {
            "bs_vi": ["L3", "L4", "L5", "M1", "N1"]
        }
    },
    "DOC": {
        "full_name": "Diesel Oxidation Catalyst",
        "description": "Oxidizes CO and HC in diesel exhaust",
        "applicable_fuels": ["Diesel"],
        "not_applicable_fuels": ["Petrol", "CNG", "LPG", "Electric"],
        "required_for": {
            "bs_vi": ["L3", "L4", "L5", "M1", "M2", "M3", "N1", "N2", "N3"]
        }
    },
    "DPF": {
        "full_name": "Diesel Particulate Filter",
        "description": "Captures and removes particulate matter from diesel exhaust",
        "applicable_fuels": ["Diesel"],
        "not_applicable_fuels": ["Petrol", "CNG", "LPG", "Electric"],
        "required_for": {
            "bs_vi": ["M1", "M2", "M3", "N1", "N2", "N3"]
        }
    },
    "SCR": {
        "full_name": "Selective Catalytic Reduction",
        "description": "Reduces NOx emissions using urea-based fluid (AdBlue)",
        "applicable_fuels": ["Diesel"],
        "not_applicable_fuels": ["Petrol", "CNG", "LPG", "Electric"],
        "required_for": {
            "bs_vi": ["M3", "N2", "N3"]
        },
        "optional_for": {
            "bs_vi": ["M1", "M2", "N1"]
        }
    },
    "EGR": {
        "full_name": "Exhaust Gas Recirculation",
        "description": "Reduces NOx by recirculating exhaust gas back into the engine",
        "applicable_fuels": ["Petrol", "Diesel", "CNG", "LPG"],
        "not_applicable_fuels": ["Electric"],
        "required_for": {
            "bs_vi": ["L5", "M1", "M2", "M3", "N1", "N2", "N3"]
        }
    },
    "EVP": {
        "full_name": "Evaporative Emission Control System",
        "description": "Prevents fuel vapors from escaping into the atmosphere",
        "applicable_fuels": ["Petrol", "LPG", "CNG"],
        "not_applicable_fuels": ["Diesel", "Electric"],
        "required_for": {
            "bs_vi": ["L3", "L4", "L5", "M1", "N1"]
        },
        "notes": "Not required for diesel vehicles as diesel fuel is less volatile and produces minimal evaporative emissions"
    },
    "OBD": {
        "full_name": "On-Board Diagnostics",
        "description": "System that monitors emissions performance and alerts when systems malfunction",
        "applicable_fuels": ["Petrol", "Diesel", "CNG", "LPG", "Hybrid"],
        "not_applicable_fuels": ["Electric"],
        "required_for": {
            "bs_vi": ["L3", "L4", "L5", "M1", "M2", "M3", "N1", "N2", "N3"]
        }
    },
    "PCV": {
        "full_name": "Positive Crankcase Ventilation",
        "description": "Prevents crankcase emissions from venting to the atmosphere",
        "applicable_fuels": ["Petrol", "Diesel", "CNG", "LPG"],
        "not_applicable_fuels": ["Electric"],
        "required_for": {
            "bs_vi": ["L3", "L4", "L5", "M1", "M2", "M3", "N1", "N2", "N3"]
        }
    }
}

# BS VI emission limits by vehicle category and fuel type
BS_VI_EMISSION_LIMITS = {
    "L2-Petrol": {
        "CO": "0.5 g/km",
        "HC+NOx": "0.35 g/km",
        "test_procedure": "IDC"
    },
    "L2-CNG": {
        "CO": "0.5 g/km",
        "HC+NOx": "0.3 g/km",
        "test_procedure": "IDC"
    },
    "L5-Petrol": {
        "CO": "0.5 g/km",
        "HC+NOx": "0.35 g/km",
        "PM": "0.0045 g/km (for DI engines)",
        "test_procedure": "WMTC"
    },
    "L5-Diesel": {
        "CO": "0.5 g/km",
        "HC+NOx": "0.3 g/km",
        "PM": "0.025 g/km",
        "test_procedure": "WMTC"
    },
    "M1-Petrol": {
        "CO": "1.0 g/km",
        "THC": "0.1 g/km",
        "NMHC": "0.068 g/km",
        "NOx": "0.06 g/km",
        "PM": "0.0045 g/km (for DI engines)",
        "test_procedure": "MIDC"
    },
    "M1-Diesel": {
        "CO": "0.5 g/km",
        "THC+NOx": "0.17 g/km",
        "NOx": "0.08 g/km",
        "PM": "0.0045 g/km",
        "PN": "6.0×10¹¹ #/km",
        "test_procedure": "MIDC"
    }
}


# Add PM (Particulate Matter) Limits to the existing knowledge base
PM_LIMITS_DATABASE = {
    # Light Commercial Vehicles (LCV) Limits
    'LCV': {
        'Diesel': {
            'General': '0.005 g/km',
            'Urban Delivery': '0.004 g/km',
            'Long Haul': '0.006 g/km'
        },
        'Petrol': {
            'General': '0.004 g/km'
        }
    },
    
    # N-Category Vehicles (Commercial Vehicles)
    'N1': {
        'Diesel': {
            'Class I (≤1305 kg)': {
                'PM Limit': '0.005 g/km',
                'Reference Standard': 'AIS 137 Rev 2'
            },
            'Class II (1305-1760 kg)': {
                'PM Limit': '0.005 g/km',
                'Reference Standard': 'AIS 137 Rev 2'
            },
            'Class III (>1760 kg)': {
                'PM Limit': '0.005 g/km',
                'Reference Standard': 'AIS 137 Rev 2'
            }
        },
        'Petrol': {
            'All Classes': {
                'PM Limit': '0.004 g/km',
                'Reference Standard': 'AIS 137 Rev 2'
            }
        }
    },
    
    # M-Category Vehicles (Passenger Vehicles)
    'M1': {
        'Diesel': {
            'Passenger Cars': {
                'PM Limit': '0.005 g/km',
                'Reference Standard': 'AIS 137 Rev 2',
                'Test Cycle': 'WMSC (Worldwide Harmonized Light Vehicle Test Cycle)'
            }
        },
        'Petrol': {
            'Passenger Cars': {
                'PM Limit': '0.004 g/km',
                'Reference Standard': 'AIS 137 Rev 2',
                'Test Cycle': 'WMSC'
            }
        }
    }
}

# Existing content remains the same, with this function added to handle PM limits
def get_pm_limits(vehicle_category, fuel_type=None):
    """
    Retrieve PM limits with comprehensive details
    
    Args:
        vehicle_category (str): Vehicle category (LCV, N1, M1)
        fuel_type (str, optional): Fuel type (Diesel, Petrol)
    
    Returns:
        dict: Detailed PM limit information
    """
    # Normalize inputs
    vehicle_category = vehicle_category.upper()
    fuel_type = fuel_type.capitalize() if fuel_type else None
    
    # Retrieve limits from PM_LIMITS_DATABASE
    category_data = PM_LIMITS_DATABASE.get(vehicle_category, {})
    
    # If no fuel type specified, return all limits for the category
    if not fuel_type:
        return category_data
    
    # Return limits for specific fuel type
    fuel_data = category_data.get(fuel_type, {})
    
    return fuel_data if fuel_data else None

# Extend existing functions to include PM limit information
def get_emission_limits(vehicle_category, fuel_type):
    """
    Extended to include PM limits along with existing emission limits
    """
    # Get base emission limits
    base_limits = BS_VI_EMISSION_LIMITS.get(f"{vehicle_category}-{fuel_type}")
    
    # Get PM limits
    pm_limits = get_pm_limits(vehicle_category, fuel_type)
    
    # Combine limits if both exist
    if base_limits and pm_limits:
        combined_limits = base_limits.copy()
        
        # Add PM limits if not already present
        if isinstance(pm_limits, dict):
            for key, value in pm_limits.items():
                if key == 'PM Limit':
                    combined_limits['PM'] = value
                elif key not in ['Reference Standard', 'Test Cycle']:
                    combined_limits[key] = value
        
        return combined_limits
    
    return base_limits or pm_limits






# Test procedures for different vehicle categories and standards
TEST_PROCEDURES = {
    "IDC": {
        "full_name": "Indian Driving Cycle",
        "applicable_to": ["L1", "L2", "L3", "L4", "L5"],
        "duration": "1180 seconds",
        "average_speed": "19.9 km/h"
    },
    "MIDC": {
        "full_name": "Modified Indian Driving Cycle",
        "applicable_to": ["M1", "M2", "N1", "N2"],
        "derived_from": "NEDC with Indian modifications",
        "phases": "Urban and Extra-Urban"
    },
    "WMTC": {
        "full_name": "World Motorcycle Test Cycle",
        "applicable_to": ["L3", "L4", "L5"],
        "phases": "3 phases (low, medium, high speed)",
        "harmonized": "Internationally harmonized test cycle"
    }
}

# Domain-specific inference rules for question answering
INFERENCE_RULES = [
    {
        "question_pattern": r"(?i).*\b(EVP|evaporative|EVAP|vapor|vapour|canister).*\b(diesel|diesel vehicle|diesel engine).*",
        "answer": "Evaporative Emission Control Systems (EVP/EVAP) are NOT required for diesel vehicles under BS VI norms. Diesel fuel is less volatile than petrol and produces minimal evaporative emissions, so these control systems are only mandated for petrol, CNG, and LPG vehicles. Diesel vehicles focus on other emission control technologies like DPF (Diesel Particulate Filter) and SCR (Selective Catalytic Reduction) to control particulate matter and NOx emissions.",
        "confidence": "high"
    },
    {
        "question_pattern": r"(?i).*\b(DPF|diesel particulate filter).*\b(petrol|gasoline|petrol vehicle).*",
        "answer": "Diesel Particulate Filters (DPF) are NOT applicable to petrol/gasoline vehicles. DPF systems are specifically designed to capture particulate matter (soot) from diesel exhaust and are required for diesel vehicles under BS VI norms. Petrol vehicles use different emission control technologies like Three-Way Catalysts (TWC) to control emissions.",
        "confidence": "high"
    },
    {
        "question_pattern": r"(?i).*\b(SCR|selective catalytic|urea|adblue).*\b(petrol|gasoline).*",
        "answer": "Selective Catalytic Reduction (SCR) systems are NOT typically used in petrol/gasoline vehicles. SCR technology is primarily designed for diesel vehicles to reduce NOx emissions and requires AdBlue (urea solution). Petrol vehicles generally use Three-Way Catalysts (TWC) to control NOx emissions and do not require AdBlue.",
        "confidence": "high"
    },
    {
        "question_pattern": r"(?i).*\b(OBD|on[ -]board diagnostic).*\b(electric|ev|battery electric).*",
        "answer": "On-Board Diagnostics (OBD) systems for emission monitoring are not required for pure electric vehicles (EVs) under BS VI norms, as EVs do not produce tailpipe emissions. However, EVs do have diagnostic systems for monitoring battery, motor, and other electrical systems, but these serve a different purpose than emission-related OBD systems required for combustion engine vehicles.",
        "confidence": "high"
    },
    {
        "question_pattern": r"(?i).*\b(emission|pollution|exhaust).*\b(test|testing|certification).*\b(electric|ev|battery electric).*",
        "answer": "Pure electric vehicles (EVs) are exempt from tailpipe emission testing as they produce zero direct emissions. EVs do not undergo the standard IDC/MIDC/WMTC emission test cycles that apply to combustion engine vehicles. However, EVs must still comply with other regulatory requirements related to safety, electromagnetic compatibility, and battery disposal.",
        "confidence": "high"
    },
    {
        "question_pattern": r"(?i).*\b(particulate matter|PM|soot).*\b(limit|standard).*\b(LCV|N1|M1).*",
        "answer": "Particulate Matter (PM) limits for BS VI standards vary by vehicle category and fuel type. Generally, diesel vehicles have stricter PM limits (around 0.005 g/km) compared to petrol vehicles (around 0.004 g/km). The specific limits depend on the vehicle's weight class, fuel type, and intended use. For precise limits, reference AIS 137 Revision 2 and CMV Rules.",
        "confidence": "high"
    },
    {
        "question_pattern": r"(?i).*\b(particulate matter|PM|soot).*\b(diesel|diesel vehicle).*",
        "answer": "Diesel vehicles have more stringent Particulate Matter (PM) control requirements under BS VI norms. Diesel Particulate Filters (DPF) are mandatory for most diesel vehicles to reduce PM emissions. Typical PM limits for diesel vehicles range from 0.004 to 0.006 g/km, depending on the vehicle category and weight class. The primary goal is to significantly reduce soot and particulate emissions compared to previous emission standards.",
        "confidence": "high"
    }
]

def get_system_compatibility(system_name, fuel_type):
    """Determine if an emission control system is compatible with a fuel type"""
    system = EMISSION_CONTROL_SYSTEMS.get(system_name.upper())
    if not system:
        return None
    
    if fuel_type in system.get("applicable_fuels", []):
        return {
            "compatible": True,
            "system": system["full_name"],
            "fuel": fuel_type,
            "description": system["description"],
            "notes": system.get("notes", "")
        }
    elif fuel_type in system.get("not_applicable_fuels", []):
        return {
            "compatible": False,
            "system": system["full_name"],
            "fuel": fuel_type,
            "description": system["description"],
            "notes": system.get("notes", "")
        }
    return None

def match_inference_rule(question):
    """Match a question to predefined inference rules to provide expert answers"""
    import re
    
    for rule in INFERENCE_RULES:
        if re.search(rule["question_pattern"], question):
            return rule["answer"]
    
    return None

def get_emission_limits(vehicle_category, fuel_type):
    """Get BS VI emission limits for a specific vehicle category and fuel type"""
    key = f"{vehicle_category}-{fuel_type}"
    return BS_VI_EMISSION_LIMITS.get(key)

def is_system_required(system_code, vehicle_category, standard="bs_vi"):
    """Check if a specific emission control system is required for a vehicle category"""
    system = EMISSION_CONTROL_SYSTEMS.get(system_code.upper())
    if not system:
        return None
    
    required_categories = system.get("required_for", {}).get(standard, [])
    optional_categories = system.get("optional_for", {}).get(standard, [])
    
    if vehicle_category in required_categories:
        return {
            "required": True,
            "status": "Required",
            "vehicle_category": vehicle_category,
            "system": system["full_name"]
        }
    elif vehicle_category in optional_categories:
        return {
            "required": False,
            "status": "Optional",
            "vehicle_category": vehicle_category,
            "system": system["full_name"]
        }
    else:
        return {
            "required": False,
            "status": "Not specified",
            "vehicle_category": vehicle_category,
            "system": system["full_name"]
        }



# Particulate Matter (PM), NOx, CO, PN emission limits under BS-VI
EMISSION_LIMITS = {
    "BS-VI": {
        "N1": {"PM": "0.005 g/km", "NOx": "0.08 g/km", "CO": "0.5 g/km", "PN": "6.0x10^11 #/km"},
        "M1": {"PM": "0.0045 g/km", "NOx": "0.06 g/km", "CO": "1.0 g/km", "PN": "6.0x10^11 #/km"},
        "N2": {"PM": "0.01 g/km", "NOx": "0.125 g/km", "CO": "1.5 g/km", "PN": "6.0x10^11 #/km"},
        "N3": {"PM": "0.01 g/km", "NOx": "0.125 g/km", "CO": "1.5 g/km", "PN": "6.0x10^11 #/km"}
    }
}

# Real Driving Emission (RDE) applicability and conformity factors
RDE_REQUIREMENTS = {
    "BS-VI": {
        "M1": {
            "applicable": True,
            "start_date": "2023-04-01",
            "conformity_factors": {"NOx": 1.43, "PN": 1.5}
        },
        "N1": {
            "applicable": True,
            "start_date": "2023-04-01",
            "conformity_factors": {"NOx": 1.43, "PN": 1.5}
        }
    }
}

# On-Board Diagnostics (OBD) stages and monitored components
OBD_REQUIREMENTS = {
    "BS-VI": {
        "OBD-I": {
            "start_date": "2020-04-01",
            "monitored_components": ["catalyst", "oxygen sensor", "misfire"]
        },
        "OBD-II": {
            "start_date": "2023-04-01",
            "monitored_components": ["NOx sensor", "PM sensor", "EGR", "fuel system"]
        }
    }
}

# Test cycles for emission testing
TEST_CYCLES = {
    "BS-VI": {
        "M1": "WLTP",
        "N1": "WHTC",
        "N2": "WHTC",
        "N3": "WHTC"
    }
}

# Emission regulation implementation dates
IMPLEMENTATION_DATES = {
    "BS-VI": {
        "all_vehicles": "2020-04-01",
        "OBD-II": "2023-04-01"
    }
}

# Source references for regulatory standards
STANDARDS_REFERENCES = {
    "BS-VI": {
        "source": "GSR 889(E)",
        "published_by": "MoRTH",
        "date": "2016-09-16"
    },
    "AIS_137": {
        "type": "Test procedure standard",
        "applies_to": ["M1", "N1", "N2", "N3"]
    }
}
