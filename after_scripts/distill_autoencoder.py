"""Train the step-64 Roformer as a distilled autoencoder."""

from absl import app

from after_scripts import train_autoencoder

FLAGS = train_autoencoder.FLAGS


def main(argv):
    # Keep every train_autoencoder flag available, while making this command a
    # convenient, correctly configured entry point for the distillation path.
    if not FLAGS["force_latent"].present:
        FLAGS.force_latent = True
    if not FLAGS["config"].present:
        FLAGS.config = ["AE_64_roformer"]
    return train_autoencoder.main(argv)


if __name__ == "__main__":
    app.run(main)
