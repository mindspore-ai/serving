import os
from mindspore_serving import worker


def export_add_net():
    from .servables.export_model import add_model
    add_model.export_net()


export_add_net()


servable_dir = os.path.abspath(".") +"/servables"


def test_model_type_mindir_success():
    worker.start_servable(servable_directory=servable_dir,
                          servable_name="add",
                          version_number=0,
                          device_id=0,
                          master_ip="127.0.0.1",
                          master_port=6500,
                          host_ip="127.0.0.1",
                          host_port=7000)
    worker.stop()


def test_model_type_mindir_error_model_type():
    try:
        worker.start_servable(servable_directory=servable_dir,
                              servable_name="add_error_model_type",
                              version_number=0,
                              device_id=0,
                              master_ip="127.0.0.1",
                              master_port=6500,
                              host_ip="127.0.0.1",
                              host_port=7000)
        worker.stop()
    except:
        return
    assert (False)


