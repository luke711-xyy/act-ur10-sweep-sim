"""Causal replay must not pin every full-resolution query image in RAM."""
import gc
import weakref

import numpy as np

from sim.act import evaluate


def test_causal_replay_releases_query_images_after_prediction():
    image = np.ones((3, 48, 64), dtype=np.float32)
    image_ref = weakref.ref(image)
    state = np.arange(36, dtype=np.float32)
    query = {'observation.images.top': image, 'observation.images.wrist': image,
             'observation.state': state, 't': 0.2, 'contact_latched': True}
    retained = evaluate._retained_query_observation(query, capture_replay=True)
    assert query['observation.images.top'] is image  # Prediction input stays intact.
    assert retained['t'] == 0.2 and retained['contact_latched'] is True
    np.testing.assert_array_equal(retained['observation.state'], state)
    del query, image
    gc.collect()
    assert image_ref() is None  # The episode's history no longer owns either camera.


def test_non_replay_library_call_keeps_its_existing_observation_contract():
    image = np.zeros((3, 48, 64), dtype=np.float32)
    query = {'observation.images.top': image, 't': 0.2}
    retained = evaluate._retained_query_observation(query, capture_replay=False)
    assert retained['observation.images.top'] is image
