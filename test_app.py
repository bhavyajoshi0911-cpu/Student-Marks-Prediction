"""
test_app.py
Comprehensive test suite verifying all routes, ML inference, and edge cases.
"""

from app import app

client = app.test_client()

def run_tests():
    print("=== 1. Testing Route / (Dashboard) ===")
    res = client.get('/')
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    assert b"Student Marks Prediction System" in res.data
    assert b"Total Records" in res.data
    assert b"Model R" in res.data
    print("  [PASS] / returned 200 and contained dashboard components.")

    print("=== 2. Testing Route /data-overview ===")
    res = client.get('/data-overview')
    assert res.status_code == 200
    assert b"Descriptive Statistical Summary" in res.data
    assert b"Full Dataset Records" in res.data
    print("  [PASS] /data-overview returned 200 and contained dataset records.")

    print("=== 3. Testing Route /model-performance ===")
    res = client.get('/model-performance')
    assert res.status_code == 200
    assert b"Mean Absolute Error" in res.data
    assert b"Matplotlib Visualizations" in res.data
    print("  [PASS] /model-performance returned 200 and contained performance cards.")

    print("=== 4. Testing Route /about ===")
    res = client.get('/about')
    assert res.status_code == 200
    assert b"Technology Stack" in res.data
    assert b"Ordinary Least Squares" in res.data
    print("  [PASS] /about returned 200.")

    print("=== 5. Testing Prediction API (Valid 6.5 hours) ===")
    res = client.post('/predict', json={'study_hours': 6.5})
    assert res.status_code == 200
    data = res.get_json()
    assert data['success'] is True
    print(f"  [PASS] Predicted for 6.5 hrs: {data['predicted_marks']} / 100 (Grade: {data['grade']})")

    print("=== 6. Testing Prediction API (Boundary 0.0 hours) ===")
    res = client.post('/predict', json={'study_hours': 0.0})
    assert res.status_code == 200
    data = res.get_json()
    assert data['success'] is True
    print(f"  [PASS] Predicted for 0.0 hrs: {data['predicted_marks']} / 100")

    print("=== 7. Testing Prediction API (Negative Hours) ===")
    res = client.post('/predict', json={'study_hours': -2})
    assert res.status_code == 400
    data = res.get_json()
    assert data['success'] is False
    print(f"  [PASS] Negative hours rejected: {data['error']}")

    print("=== 8. Testing Prediction API (> 24 Hours) ===")
    res = client.post('/predict', json={'study_hours': 30})
    assert res.status_code == 400
    data = res.get_json()
    assert data['success'] is False
    print(f"  [PASS] >24 hours rejected: {data['error']}")

    print("=== 9. Testing Prediction API (Empty Input) ===")
    res = client.post('/predict', json={'study_hours': ''})
    assert res.status_code == 400
    data = res.get_json()
    assert data['success'] is False
    print(f"  [PASS] Empty input rejected: {data['error']}")

    print("=== 10. Testing Prediction API (Non-numeric Input) ===")
    res = client.post('/predict', json={'study_hours': 'abc'})
    assert res.status_code == 400
    data = res.get_json()
    assert data['success'] is False
    print(f"  [PASS] Non-numeric input rejected: {data['error']}")

    print("=== 11. Testing /api/chart-data ===")
    res = client.get('/api/chart-data')
    assert res.status_code == 200
    data = res.get_json()
    assert len(data['scatter_data']) == 100
    print(f"  [PASS] /api/chart-data returned {len(data['scatter_data'])} scatter points.")

    print("=== 12. Testing /download-dataset ===")
    res = client.get('/download-dataset')
    assert res.status_code == 200
    assert b"Study_Hours" in res.data
    print("  [PASS] /download-dataset streamed CSV dataset.")

    print("=== 13. Testing 404 Error Handler ===")
    res = client.get('/nonexistent-test-page')
    assert res.status_code == 404
    assert b"Page Not Found" in res.data
    print("  [PASS] 404 handler returned custom 404 page.")

    print("\n==============================================")
    print(">>> ALL 13 TEST SUITES PASSED SUCCESSFULLY! <<<")
    print("==============================================")


if __name__ == "__main__":
    run_tests()
