"""
ML Models 테스트 - SklearnModel, PyTorchModel
외부 라이브러리(sklearn, torch, joblib)를 Mock하여 테스트
"""

import hashlib
import hmac
import pickle
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, mock_open, patch

import pytest

# ---------------------------------------------------------------------------
# SklearnModel 테스트
# ---------------------------------------------------------------------------


class TestSklearnModel:
    """SklearnModel 클래스 테스트"""

    def _make_model(self, estimator=None):
        """SklearnModel 인스턴스 생성 헬퍼"""
        from beanllm.infrastructure.ml.sklearn_model import SklearnModel

        return SklearnModel(model=estimator)

    # --- 초기화 ---

    def test_init_no_model(self):
        """모델 없이 초기화"""
        from beanllm.infrastructure.ml.sklearn_model import SklearnModel

        m = SklearnModel()
        assert m.model is None

    def test_init_with_estimator(self):
        """estimator로 초기화"""
        estimator = MagicMock()
        m = self._make_model(estimator)
        assert m.model is estimator

    def test_from_estimator_classmethod(self):
        """from_estimator 클래스 메서드"""
        from beanllm.infrastructure.ml.sklearn_model import SklearnModel

        estimator = MagicMock()
        m = SklearnModel.from_estimator(estimator)
        assert m.model is estimator

    # --- predict ---

    def test_predict_no_model_raises(self):
        """모델 없이 predict 호출 시 ValueError"""
        m = self._make_model()
        with pytest.raises(ValueError, match="Model not loaded"):
            m.predict([[1, 2, 3]])

    def test_predict_calls_model(self):
        """predict가 내부 모델의 predict를 호출하는지 확인"""
        estimator = MagicMock()
        estimator.predict.return_value = [0, 1, 0]
        m = self._make_model(estimator)
        result = m.predict([[1, 2], [3, 4]])
        estimator.predict.assert_called_once()
        assert result == [0, 1, 0]

    def test_predict_passes_kwargs(self):
        """predict에 kwargs가 전달되는지 확인"""
        estimator = MagicMock()
        estimator.predict.return_value = [1]
        m = self._make_model(estimator)
        m.predict([[1]], check_input=False)
        estimator.predict.assert_called_once_with([[1]], check_input=False)

    # --- predict_proba ---

    def test_predict_proba_no_model_raises(self):
        """모델 없이 predict_proba 호출 시 ValueError"""
        m = self._make_model()
        with pytest.raises(ValueError, match="Model not loaded"):
            m.predict_proba([[1, 2]])

    def test_predict_proba_no_support_raises(self):
        """predict_proba 미지원 모델 시 AttributeError"""
        estimator = MagicMock(spec=[])  # spec=[] → no predict_proba attribute
        m = self._make_model(estimator)
        with pytest.raises(AttributeError, match="predict_proba"):
            m.predict_proba([[1, 2]])

    def test_predict_proba_success(self):
        """predict_proba 정상 호출"""
        estimator = MagicMock()
        estimator.predict_proba.return_value = [[0.3, 0.7]]
        m = self._make_model(estimator)
        result = m.predict_proba([[1, 2]])
        assert result == [[0.3, 0.7]]

    # --- fit ---

    def test_fit_no_model_raises(self):
        """모델 없이 fit 호출 시 ValueError"""
        m = self._make_model()
        with pytest.raises(ValueError, match="Model not initialized"):
            m.fit([[1, 2]], [0])

    def test_fit_calls_model(self):
        """fit이 내부 모델의 fit을 호출하는지 확인"""
        estimator = MagicMock()
        m = self._make_model(estimator)
        m.fit([[1, 2], [3, 4]], [0, 1])
        estimator.fit.assert_called_once_with([[1, 2], [3, 4]], [0, 1])

    def test_fit_passes_kwargs(self):
        """fit에 kwargs가 전달되는지 확인"""
        estimator = MagicMock()
        m = self._make_model(estimator)
        m.fit([[1]], [0], sample_weight=[1.0])
        estimator.fit.assert_called_once_with([[1]], [0], sample_weight=[1.0])

    # --- save (with joblib) ---

    def test_save_no_model_raises(self):
        """모델 없이 save 호출 시 ValueError"""
        m = self._make_model()
        with pytest.raises(ValueError, match="No model to save"):
            m.save("/tmp/model.pkl")

    def test_save_with_joblib(self, tmp_path):
        """joblib을 사용한 저장 테스트"""
        estimator = MagicMock()
        m = self._make_model(estimator)
        save_path = tmp_path / "model.pkl"

        mock_joblib = MagicMock()
        # joblib.dump는 파일 쓰기를 실제로 수행 — 여기서는 내용이 빈 파일로 대체하기 위해
        # 실제 파일 쓰기는 sign=False 로 비활성화하고 joblib을 patch
        with patch.dict(sys.modules, {"joblib": mock_joblib}):
            # sign=False로 서명 없이 저장
            m.save(save_path, use_joblib=True, sign=False)

        mock_joblib.dump.assert_called_once_with(estimator, str(save_path))

    def test_save_with_pickle(self, tmp_path):
        """pickle을 사용한 저장 테스트"""
        estimator = MagicMock()
        m = self._make_model(estimator)
        save_path = tmp_path / "model.pkl"

        # sign=False 로 서명 비활성화
        with patch("builtins.open", mock_open()):
            with patch("pickle.dump") as mock_dump:
                m.save(save_path, use_joblib=False, sign=False)
                mock_dump.assert_called_once()

    def test_save_creates_signature_file(self, tmp_path):
        """서명 파일이 생성되는지 확인"""
        import os

        estimator = MagicMock()
        m = self._make_model(estimator)
        save_path = tmp_path / "model.pkl"

        mock_joblib = MagicMock()

        # joblib.dump는 실제로 파일을 생성해야 서명이 가능하므로
        # 실제 파일을 빈 내용으로 미리 생성
        save_path.write_bytes(b"fake_model_bytes")

        with patch.dict(sys.modules, {"joblib": mock_joblib}):
            # sign=True
            m.save(save_path, use_joblib=True, sign=True)

        sig_path = Path(f"{save_path}.sig")
        assert sig_path.exists()
        assert len(sig_path.read_text()) == 64  # sha256 hex digest = 64 chars

    def test_save_fallback_to_pickle_when_joblib_unavailable(self, tmp_path):
        """joblib 없을 때 pickle 폴백 테스트"""
        estimator = MagicMock()
        m = self._make_model(estimator)
        save_path = tmp_path / "model.pkl"

        # joblib.dump가 ImportError를 발생시키는 상황을 시뮬레이션
        # sys.modules에 joblib이 없도록 하고 ImportError를 발생시키는 모듈 등록
        mock_joblib = MagicMock()
        mock_joblib.dump.side_effect = ImportError("joblib not available")

        with patch.dict(sys.modules, {"joblib": mock_joblib}):
            with patch("pickle.dump") as mock_pickle_dump:
                m.save(save_path, use_joblib=True, sign=False)
                mock_pickle_dump.assert_called_once()

    # --- load ---

    def test_load_missing_signature_raises(self, tmp_path):
        """서명 파일 없으면 ValueError"""
        from beanllm.infrastructure.ml.sklearn_model import SklearnModel

        m = SklearnModel()
        model_path = tmp_path / "model.pkl"
        model_path.write_bytes(b"data")
        # sig 파일 없음

        with pytest.raises(ValueError, match="Signature file"):
            m.load(model_path, verify_signature=True)

    def test_load_wrong_signature_raises(self, tmp_path):
        """잘못된 서명이면 ValueError"""
        from beanllm.infrastructure.ml.sklearn_model import SklearnModel

        m = SklearnModel()
        model_path = tmp_path / "model.pkl"
        model_path.write_bytes(b"data")
        sig_path = tmp_path / "model.pkl.sig"
        sig_path.write_text("wrong_signature")

        with pytest.raises(ValueError, match="Signature verification failed"):
            m.load(model_path, verify_signature=True)

    def test_load_with_valid_signature(self, tmp_path):
        """올바른 서명으로 로드 성공 (joblib mock 사용)"""
        from beanllm.infrastructure.ml.sklearn_model import MODEL_SIGNATURE_KEY, SklearnModel

        # 임의 바이트로 파일 생성 후 서명 계산
        model_bytes = b"fake_model_data_for_signature_test"
        model_path = tmp_path / "model.pkl"
        model_path.write_bytes(model_bytes)

        sig = hmac.new(MODEL_SIGNATURE_KEY.encode(), model_bytes, hashlib.sha256).hexdigest()
        sig_path = tmp_path / "model.pkl.sig"
        sig_path.write_text(sig)

        fake_loaded = object()  # simple sentinel
        mock_joblib = MagicMock()
        mock_joblib.load.return_value = fake_loaded

        m = SklearnModel()
        with patch.dict(sys.modules, {"joblib": mock_joblib}):
            m.load(model_path, verify_signature=True)

        assert m.model is fake_loaded

    def test_load_without_signature_verification(self, tmp_path):
        """서명 검증 없이 로드 (joblib mock 사용)"""
        from beanllm.infrastructure.ml.sklearn_model import SklearnModel

        model_path = tmp_path / "model.pkl"
        model_path.write_bytes(b"any_data")

        fake_loaded = object()
        mock_joblib = MagicMock()
        mock_joblib.load.return_value = fake_loaded

        m = SklearnModel()
        with patch.dict(sys.modules, {"joblib": mock_joblib}):
            m.load(model_path, verify_signature=False)

        assert m.model is fake_loaded

    def test_load_with_joblib(self, tmp_path):
        """joblib으로 로드"""
        from beanllm.infrastructure.ml.sklearn_model import SklearnModel

        model_path = tmp_path / "model.pkl"
        model_path.write_bytes(b"fake")

        fake_estimator = MagicMock()
        mock_joblib = MagicMock()
        mock_joblib.load.return_value = fake_estimator

        m = SklearnModel()
        with patch.dict(sys.modules, {"joblib": mock_joblib}):
            m.load(model_path, verify_signature=False)

        assert m.model is fake_estimator
        mock_joblib.load.assert_called_once()

    def test_load_failure_raises(self, tmp_path):
        """로드 실패 시 ValueError"""
        from beanllm.infrastructure.ml.sklearn_model import SklearnModel

        model_path = tmp_path / "model.pkl"
        model_path.write_bytes(b"not_valid_pickle")

        m = SklearnModel()
        mock_joblib = MagicMock()
        mock_joblib.load.side_effect = Exception("joblib failed")
        with patch.dict(sys.modules, {"joblib": mock_joblib}):
            with pytest.raises(ValueError, match="Failed to load model"):
                m.load(model_path, verify_signature=False)

    # --- from_pickle classmethod (just delegates to load) ---

    def test_from_pickle_no_sig_file(self, tmp_path):
        """from_pickle은 서명 검증 수행 → sig 없으면 ValueError"""
        from beanllm.infrastructure.ml.sklearn_model import SklearnModel

        model_path = tmp_path / "model.pkl"
        model_path.write_bytes(b"data")

        with pytest.raises(ValueError, match="Signature file"):
            SklearnModel.from_pickle(model_path)


# ---------------------------------------------------------------------------
# PyTorchModel 테스트
# ---------------------------------------------------------------------------


class TestPyTorchModel:
    """PyTorchModel 클래스 테스트"""

    def _make_torch_mock(self):
        """torch 모듈 mock 생성"""
        torch_mock = MagicMock()
        tensor_mock = MagicMock()
        tensor_mock.cpu.return_value = tensor_mock
        tensor_mock.numpy.return_value = [0.1, 0.9]
        torch_mock.Tensor = type("Tensor", (), {})
        torch_mock.from_numpy.return_value = tensor_mock
        torch_mock.no_grad.return_value.__enter__ = MagicMock(return_value=None)
        torch_mock.no_grad.return_value.__exit__ = MagicMock(return_value=False)
        torch_mock.cuda.is_available.return_value = False
        torch_mock.load.return_value = MagicMock()
        torch_mock.save = MagicMock()
        return torch_mock

    # --- 초기화 ---

    def test_init_no_model(self):
        """기본 초기화"""
        from beanllm.infrastructure.ml.pytorch_model import PyTorchModel

        m = PyTorchModel()
        assert m.model is None
        assert m.device in ("cuda", "cpu")

    def test_init_device_cpu_when_cuda_unavailable(self):
        """CUDA 미사용 시 device=cpu"""
        torch_mock = MagicMock()
        torch_mock.cuda.is_available.return_value = False
        with patch.dict(sys.modules, {"torch": torch_mock}):
            from beanllm.infrastructure.ml.pytorch_model import PyTorchModel

            m = PyTorchModel()
            assert m.device == "cpu"

    def test_init_device_cuda_when_available(self):
        """CUDA 사용 가능 시 device=cuda"""
        torch_mock = MagicMock()
        torch_mock.cuda.is_available.return_value = True
        with patch.dict(sys.modules, {"torch": torch_mock}):
            from beanllm.infrastructure.ml.pytorch_model import PyTorchModel

            m = PyTorchModel()
            assert m.device == "cuda"

    def test_init_explicit_device(self):
        """명시적 device 설정"""
        from beanllm.infrastructure.ml.pytorch_model import PyTorchModel

        m = PyTorchModel(device="cpu")
        assert m.device == "cpu"

    def test_init_with_model(self):
        """model 인자로 초기화"""
        from beanllm.infrastructure.ml.pytorch_model import PyTorchModel

        fake_model = MagicMock()
        m = PyTorchModel(model=fake_model, device="cpu")
        assert m.model is fake_model

    # --- load ---

    def test_load_raises_without_torch(self, tmp_path):
        """torch 없으면 ImportError"""
        from beanllm.infrastructure.ml.pytorch_model import PyTorchModel

        model_path = tmp_path / "model.pth"
        model_path.write_bytes(b"fake")

        m = PyTorchModel(device="cpu")
        saved_torch = sys.modules.get("torch")
        sys.modules["torch"] = None  # type: ignore[assignment]
        try:
            with pytest.raises((ImportError, Exception)):
                m.load(model_path)
        finally:
            if saved_torch is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = saved_torch

    def test_load_full_model(self, tmp_path):
        """전체 모델이 저장된 체크포인트 로드"""
        from beanllm.infrastructure.ml.pytorch_model import PyTorchModel

        fake_model = MagicMock()
        fake_model.to.return_value = fake_model
        torch_mock = self._make_torch_mock()
        torch_mock.load.return_value = fake_model  # not a dict with state_dict

        model_path = tmp_path / "model.pth"
        model_path.write_bytes(b"fake")

        with patch.dict(sys.modules, {"torch": torch_mock}):
            m = PyTorchModel(device="cpu")
            m.load(model_path)
            assert m.model is fake_model

    def test_load_state_dict_checkpoint_no_arch_raises(self, tmp_path):
        """state_dict 체크포인트인데 모델 아키텍처 없으면 ValueError"""
        from beanllm.infrastructure.ml.pytorch_model import PyTorchModel

        torch_mock = self._make_torch_mock()
        torch_mock.load.return_value = {"model_state_dict": {"layer.weight": MagicMock()}}

        model_path = tmp_path / "model.pth"
        model_path.write_bytes(b"fake")

        with patch.dict(sys.modules, {"torch": torch_mock}):
            m = PyTorchModel(device="cpu")
            with pytest.raises(ValueError, match="Model architecture not provided"):
                m.load(model_path)

    def test_load_state_dict_checkpoint_with_model(self, tmp_path):
        """state_dict 체크포인트 + 모델 아키텍처 제공"""
        from beanllm.infrastructure.ml.pytorch_model import PyTorchModel

        state_dict = {"layer.weight": MagicMock()}
        torch_mock = self._make_torch_mock()
        torch_mock.load.return_value = {"model_state_dict": state_dict}

        fake_arch = MagicMock()
        fake_arch.to.return_value = fake_arch

        model_path = tmp_path / "model.pth"
        model_path.write_bytes(b"fake")

        with patch.dict(sys.modules, {"torch": torch_mock}):
            m = PyTorchModel(model=fake_arch, device="cpu")
            m.load(model_path)
            fake_arch.load_state_dict.assert_called_once_with(state_dict)

    # --- predict ---

    def test_predict_no_model_raises(self):
        """모델 없이 predict → ValueError"""
        from beanllm.infrastructure.ml.pytorch_model import PyTorchModel

        torch_mock = self._make_torch_mock()
        with patch.dict(sys.modules, {"torch": torch_mock}):
            m = PyTorchModel(device="cpu")
            with pytest.raises(ValueError, match="Model not loaded"):
                m.predict([1, 2, 3])

    def test_predict_with_list_input(self):
        """리스트 입력으로 predict"""
        from beanllm.infrastructure.ml.pytorch_model import PyTorchModel

        output_tensor = MagicMock()
        output_tensor.cpu.return_value = output_tensor
        output_tensor.numpy.return_value = [0.5]

        fake_model = MagicMock()
        fake_model.return_value = output_tensor
        fake_model.to.return_value = fake_model

        torch_mock = self._make_torch_mock()
        # Make isinstance(outputs, torch.Tensor) True
        torch_mock.Tensor = type(output_tensor)
        # Make isinstance(outputs, torch.Tensor) return True
        # by making outputs itself a tensor mock
        torch_mock.no_grad.return_value.__enter__ = MagicMock(return_value=None)
        torch_mock.no_grad.return_value.__exit__ = MagicMock(return_value=False)

        with patch.dict(sys.modules, {"torch": torch_mock}):
            m = PyTorchModel(model=fake_model, device="cpu")
            m.predict([1.0, 2.0])
            fake_model.eval.assert_called()

    def test_predict_raises_without_torch(self):
        """torch 없으면 ImportError"""
        from beanllm.infrastructure.ml.pytorch_model import PyTorchModel

        fake_model = MagicMock()
        m = PyTorchModel.__new__(PyTorchModel)
        m.model = fake_model
        m.device = "cpu"
        m.model_path = None

        saved = sys.modules.get("torch")
        sys.modules["torch"] = None  # type: ignore[assignment]
        try:
            with pytest.raises((ImportError, Exception)):
                m.predict([1.0])
        finally:
            if saved is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = saved

    # --- save ---

    def test_save_no_model_raises(self):
        """모델 없이 save → ValueError"""
        from beanllm.infrastructure.ml.pytorch_model import PyTorchModel

        torch_mock = self._make_torch_mock()
        with patch.dict(sys.modules, {"torch": torch_mock}):
            m = PyTorchModel(device="cpu")
            with pytest.raises(ValueError, match="No model to save"):
                m.save("/tmp/model.pth")

    def test_save_state_dict_only(self, tmp_path):
        """save_full_model=False → state_dict 저장"""
        from beanllm.infrastructure.ml.pytorch_model import PyTorchModel

        torch_mock = self._make_torch_mock()
        fake_model = MagicMock()
        fake_model.state_dict.return_value = {"w": MagicMock()}

        save_path = tmp_path / "model.pth"
        with patch.dict(sys.modules, {"torch": torch_mock}):
            m = PyTorchModel(model=fake_model, device="cpu")
            m.save(save_path, save_full_model=False)
            torch_mock.save.assert_called_once()
            call_args = torch_mock.save.call_args[0]
            assert "model_state_dict" in call_args[0]

    def test_save_full_model(self, tmp_path):
        """save_full_model=True → 전체 모델 저장"""
        from beanllm.infrastructure.ml.pytorch_model import PyTorchModel

        torch_mock = self._make_torch_mock()
        fake_model = MagicMock()
        save_path = tmp_path / "model.pth"

        with patch.dict(sys.modules, {"torch": torch_mock}):
            m = PyTorchModel(model=fake_model, device="cpu")
            m.save(save_path, save_full_model=True)
            torch_mock.save.assert_called_once_with(fake_model, str(save_path))

    def test_save_raises_without_torch(self, tmp_path):
        """torch 없으면 ImportError"""
        from beanllm.infrastructure.ml.pytorch_model import PyTorchModel

        fake_model = MagicMock()
        m = PyTorchModel.__new__(PyTorchModel)
        m.model = fake_model
        m.device = "cpu"
        m.model_path = None

        save_path = tmp_path / "model.pth"
        saved = sys.modules.get("torch")
        sys.modules["torch"] = None  # type: ignore[assignment]
        try:
            with pytest.raises((ImportError, Exception)):
                m.save(save_path)
        finally:
            if saved is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = saved

    # --- eval_mode / train_mode ---

    def test_eval_mode_calls_eval(self):
        """eval_mode()가 모델의 eval() 호출"""
        from beanllm.infrastructure.ml.pytorch_model import PyTorchModel

        fake_model = MagicMock()
        m = PyTorchModel.__new__(PyTorchModel)
        m.model = fake_model
        m.device = "cpu"
        m.model_path = None
        m.eval_mode()
        fake_model.eval.assert_called_once()

    def test_train_mode_calls_train(self):
        """train_mode()가 모델의 train() 호출"""
        from beanllm.infrastructure.ml.pytorch_model import PyTorchModel

        fake_model = MagicMock()
        m = PyTorchModel.__new__(PyTorchModel)
        m.model = fake_model
        m.device = "cpu"
        m.model_path = None
        m.train_mode()
        fake_model.train.assert_called_once()

    def test_eval_mode_no_model(self):
        """모델 없으면 eval_mode는 아무것도 안 함"""
        from beanllm.infrastructure.ml.pytorch_model import PyTorchModel

        m = PyTorchModel.__new__(PyTorchModel)
        m.model = None
        m.device = "cpu"
        m.model_path = None
        m.eval_mode()  # should not raise

    # --- from_checkpoint classmethods ---

    def test_from_checkpoint_creates_instance(self, tmp_path):
        """from_checkpoint는 PyTorchModel 인스턴스 반환"""
        from beanllm.infrastructure.ml.pytorch_model import PyTorchModel

        torch_mock = self._make_torch_mock()
        fake_checkpoint = MagicMock()
        fake_checkpoint.to.return_value = fake_checkpoint
        torch_mock.load.return_value = fake_checkpoint

        model_path = tmp_path / "model.pth"
        model_path.write_bytes(b"fake")

        with patch.dict(sys.modules, {"torch": torch_mock}):
            instance = PyTorchModel.from_checkpoint(model_path, device="cpu")
            assert isinstance(instance, PyTorchModel)

    def test_from_checkpoint_with_model(self, tmp_path):
        """from_checkpoint_with_model은 아키텍처와 함께 로드"""
        from beanllm.infrastructure.ml.pytorch_model import PyTorchModel

        state_dict = {"w": MagicMock()}
        torch_mock = self._make_torch_mock()
        torch_mock.load.return_value = {"model_state_dict": state_dict}

        arch = MagicMock()
        arch.to.return_value = arch

        model_path = tmp_path / "model.pth"
        model_path.write_bytes(b"fake")

        with patch.dict(sys.modules, {"torch": torch_mock}):
            instance = PyTorchModel.from_checkpoint_with_model(model_path, arch, device="cpu")
            assert isinstance(instance, PyTorchModel)
            arch.load_state_dict.assert_called_once_with(state_dict)

    # --- _is_cuda_available ---

    def test_is_cuda_available_true(self):
        """torch.cuda.is_available() = True"""
        from beanllm.infrastructure.ml.pytorch_model import PyTorchModel

        torch_mock = MagicMock()
        torch_mock.cuda.is_available.return_value = True
        with patch.dict(sys.modules, {"torch": torch_mock}):
            m = PyTorchModel.__new__(PyTorchModel)
            assert m._is_cuda_available() is True

    def test_is_cuda_available_false_on_import_error(self):
        """torch ImportError 시 False 반환"""
        from beanllm.infrastructure.ml.pytorch_model import PyTorchModel

        m = PyTorchModel.__new__(PyTorchModel)
        saved = sys.modules.get("torch")
        sys.modules["torch"] = None  # type: ignore[assignment]
        try:
            result = m._is_cuda_available()
            assert result is False
        finally:
            if saved is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = saved
