"""
Firestore data serialization utilities.

Firestore has limitations on deeply nested array structures. These utilities
handle serialization of GeoJSON coordinates and other nested data to/from
JSON strings for Firestore storage.
"""

from typing import Any


def serialize_coordinates(data: dict[str, Any]) -> dict[str, Any]:
    """
    Serialize GeoJSON coordinates in a data structure to JSON strings.

    Recursively finds geometry objects with coordinates and converts the
    coordinates arrays to JSON strings for Firestore storage.

    Args:
        data: Dictionary potentially containing GeoJSON geometry objects.

    Returns:
        Modified dictionary with coordinates serialized as JSON strings.
    """
    raise NotImplementedError("TODO: Implement serialize_coordinates")


def deserialize_coordinates(data: dict[str, Any] | None) -> dict[str, Any] | None:
    """
    Deserialize GeoJSON coordinates from JSON strings back to arrays.

    Reverses the serialization done by serialize_coordinates.

    Args:
        data: Dictionary potentially containing serialized coordinates.

    Returns:
        Modified dictionary with coordinates deserialized to arrays,
        or None if input is None.
    """
    raise NotImplementedError("TODO: Implement deserialize_coordinates")


def serialize_domain_coordinates(domain_data: dict[str, Any]) -> dict[str, Any]:
    """
    Serialize domain geometry coordinates for Firestore storage.

    Handles the specific structure of domain documents where geometry
    coordinates need to be stored as JSON strings.

    Args:
        domain_data: Domain document data with geometry field.

    Returns:
        Modified domain data with serialized coordinates.
    """
    raise NotImplementedError("TODO: Implement serialize_domain_coordinates")


def deserialize_domain_coordinates(
    domain_data: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """
    Deserialize domain geometry coordinates from Firestore.

    Args:
        domain_data: Domain document data from Firestore.

    Returns:
        Modified domain data with deserialized coordinates,
        or None if input is None.
    """
    raise NotImplementedError("TODO: Implement deserialize_domain_coordinates")


def serialize_spatial_conditions(data: dict[str, Any]) -> dict[str, Any]:
    """
    Serialize spatial condition geometry coordinates in grid modification data.

    Handles the specific structure of surface grid modifications where
    spatial conditions contain GeoJSON geometries.

    Args:
        data: Grid data potentially containing modifications with spatial conditions.

    Returns:
        Modified data with spatial condition coordinates serialized.
    """
    raise NotImplementedError("TODO: Implement serialize_spatial_conditions")


def deserialize_spatial_conditions(
    data: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """
    Deserialize spatial condition geometry coordinates.

    Args:
        data: Grid data from Firestore.

    Returns:
        Modified data with spatial condition coordinates deserialized,
        or None if input is None.
    """
    raise NotImplementedError("TODO: Implement deserialize_spatial_conditions")
