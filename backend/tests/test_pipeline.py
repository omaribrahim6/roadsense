"""Basic tests for roadsense pipeline"""

import pytest
import sys
from pathlib import Path

#add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSeverity:
    """tests for severity calculation"""
    
    def test_calculate_severity_pothole(self):
        from pipeline.inference import RawDetection
        from pipeline.severity import calculate_severity
        
        detection = RawDetection(
            frame_path="test.jpg",
            timestamp=0.0,
            damage_type="pothole",
            confidence=0.9,
            bbox={"x1": 100, "y1": 100, "x2": 300, "y2": 300},
            frame_width=1920,
            frame_height=1080,
        )
        
        severity = calculate_severity(detection)
        
        assert 0.0 <= severity <= 1.0
        assert severity > 0.5  #pothole with high confidence should be severe
    
    def test_severity_increases_with_area(self):
        from pipeline.inference import RawDetection
        from pipeline.severity import calculate_severity
        
        #small damage
        small = RawDetection(
            frame_path="test.jpg",
            timestamp=0.0,
            damage_type="crack",
            confidence=0.8,
            bbox={"x1": 100, "y1": 100, "x2": 150, "y2": 150},
            frame_width=1920,
            frame_height=1080,
        )
        
        #large damage
        large = RawDetection(
            frame_path="test.jpg",
            timestamp=0.0,
            damage_type="crack",
            confidence=0.8,
            bbox={"x1": 100, "y1": 100, "x2": 500, "y2": 500},
            frame_width=1920,
            frame_height=1080,
        )
        
        assert calculate_severity(large) > calculate_severity(small)


class TestGPS:
    """tests for gps resolution"""
    
    def test_simulate_gps_start(self):
        from pipeline.gps import simulate_gps, DEMO_ROUTE_KUWAIT
        
        lat, lng = simulate_gps(0.0, DEMO_ROUTE_KUWAIT, 60.0)
        
        #should be at start of route
        assert lat == DEMO_ROUTE_KUWAIT[0][0]
        assert lng == DEMO_ROUTE_KUWAIT[0][1]
    
    def test_simulate_gps_end(self):
        from pipeline.gps import simulate_gps, DEMO_ROUTE_KUWAIT
        
        lat, lng = simulate_gps(60.0, DEMO_ROUTE_KUWAIT, 60.0)
        
        #should be at end of route
        assert lat == DEMO_ROUTE_KUWAIT[-1][0]
        assert lng == DEMO_ROUTE_KUWAIT[-1][1]
    
    def test_interpolate_midpoint(self):
        from pipeline.gps import interpolate_route
        
        route = [(0.0, 0.0), (10.0, 10.0)]
        
        lat, lng = interpolate_route(5.0, route, 10.0)
        
        #should be at midpoint
        assert abs(lat - 5.0) < 0.01
        assert abs(lng - 5.0) < 0.01


class TestDedup:
    """tests for deduplication"""
    
    def test_deduplicate_same_location(self):
        from pipeline.dedup import deduplicate, EnrichedDetection
        
        detections = [
            EnrichedDetection(
                frame_path="frame1.jpg",
                timestamp=0.0,
                damage_type="pothole",
                confidence=0.9,
                severity=0.8,
                bbox={"x1": 100, "y1": 100, "x2": 200, "y2": 200},
                lat=29.3759,
                lng=47.9774,
                frame_width=1920,
                frame_height=1080,
            ),
            EnrichedDetection(
                frame_path="frame2.jpg",
                timestamp=1.0,
                damage_type="pothole",
                confidence=0.85,
                severity=0.75,
                bbox={"x1": 110, "y1": 110, "x2": 210, "y2": 210},
                lat=29.3759,  #same location
                lng=47.9774,
                frame_width=1920,
                frame_height=1080,
            ),
        ]
        
        result = deduplicate(detections, radius_meters=10.0)
        
        #should merge into one
        assert len(result) == 1
        #should keep higher confidence
        assert result[0].confidence == 0.9
        #should track frequency
        assert result[0].frequency == 2
    
    def test_deduplicate_different_types(self):
        from pipeline.dedup import deduplicate, EnrichedDetection
        
        detections = [
            EnrichedDetection(
                frame_path="frame1.jpg",
                timestamp=0.0,
                damage_type="pothole",
                confidence=0.9,
                severity=0.8,
                bbox={"x1": 100, "y1": 100, "x2": 200, "y2": 200},
                lat=29.3759,
                lng=47.9774,
                frame_width=1920,
                frame_height=1080,
            ),
            EnrichedDetection(
                frame_path="frame1.jpg",
                timestamp=0.0,
                damage_type="crack",  #different type
                confidence=0.85,
                severity=0.75,
                bbox={"x1": 110, "y1": 110, "x2": 210, "y2": 210},
                lat=29.3759,
                lng=47.9774,
                frame_width=1920,
                frame_height=1080,
            ),
        ]
        
        result = deduplicate(detections, radius_meters=10.0)
        
        #should NOT merge - different types
        assert len(result) == 2


class TestExtractor:
    """tests for extractor utilities"""
    
    def test_calculate_distance_meters(self):
        from pipeline.extractor import calculate_distance_meters
        
        #two points ~111km apart (1 degree latitude)
        point1 = (0.0, 0.0)
        point2 = (1.0, 0.0)
        
        distance = calculate_distance_meters(point1, point2)
        
        #should be approximately 111km
        assert 110000 < distance < 112000
    
    def test_calculate_distance_same_point(self):
        from pipeline.extractor import calculate_distance_meters
        
        point = (29.3759, 47.9774)
        
        distance = calculate_distance_meters(point, point)
        
        #same point should be 0
        assert distance == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
