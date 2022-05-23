import numpy as np
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
import torch as th
from ignite.metrics import Recall, Precision, Accuracy
from ignite.engine import *


def calculate_activation_statistics(images, model, dims=4096):
  act = np.empty((len(images), dims))
  device = th.device('cuda' if th.cuda.is_available() else 'cpu')
  batch = images.to(device)
  pred = model(batch)

  # If model output is not scalar, apply global spatial average pooling.
  # This happens if you choose a dimensionality not equal 2048.
  if pred.size(2) != 1 or pred.size(3) != 1:
    pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

  act = pred.cpu().data.numpy().reshape(pred.size(0), -1)

  mu = np.mean(act, axis=0)
  sigma = np.cov(act, rowvar=False)
  return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
  """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2  =  ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    """

  mu1 = np.atleast_1d(mu1)
  mu2 = np.atleast_1d(mu2)

  sigma1 = np.atleast_2d(sigma1)
  sigma2 = np.atleast_2d(sigma2)

  assert mu1.shape == mu2.shape, 'Training and test mean vectors have different lengths'
  assert sigma1.shape == sigma2.shape, 'Training and test covariances have different dimensions'

  diff = mu1 - mu2

  covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
  if not np.isfinite(covmean).all():
    msg = 'fid calculation produces singular product; adding %s to diagonal of cov estimates' % eps
    print(msg)
    offset = np.eye(sigma1.shape[0]) * eps
    covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

  if np.iscomplexobj(covmean):
    if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
      m = np.max(np.abs(covmean.imag))
      raise ValueError('Imaginary component {}'.format(m))
    covmean = covmean.real

  tr_covmean = np.trace(covmean)

  return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_fretchet(images_real, images_fake, model):
  model.eval()
  with th.no_grad():
    mu_1, std_1 = calculate_activation_statistics(images_real, model)
    mu_2, std_2 = calculate_activation_statistics(images_fake, model)

    """get fretched distance"""
    fid_value = calculate_frechet_distance(mu_1, std_1, mu_2, std_2)

  model.train()
  return fid_value


def eval_step(engine, batch):
  return batch


default_evaluator = Engine(eval_step)


def calculate_precision_recall(real_image, fake_image):
  recall = Recall(average=False)
  precision = Precision(average=False)
  accuracy = Accuracy()
  F1 = (precision * recall * 2 / (precision + recall)).mean()

  recall.attach(default_evaluator, "recall")
  precision.attach(default_evaluator, "precision")
  accuracy.attach(default_evaluator, "accuracy")
  F1.attach(default_evaluator, "f1")

  state = default_evaluator.run([[fake_image, real_image]])
  return state.metrics


def calculate_evaluation_metrics(images_real, images_fake, model):
  fid = calculate_fretchet(images_real, images_fake, model)
  metrics = calculate_precision_recall(real_image=images_real, fake_image=images_fake)
  return fid, metrics["recall"], metrics["precision"]
