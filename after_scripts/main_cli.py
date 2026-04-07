import sys

from absl import app

AVAILABLE_SCRIPTS = [
    'train', 'prepare_dataset', 'update_dataset', 'train_autoencoder',
    'export_autoencoder', 'export', 'export_midi', "export_clock"
]


def help():
    print(f"""usage: after [ {' | '.join(AVAILABLE_SCRIPTS)} ]

positional arguments:
  command     Command to launch with after.
""")
    exit()


def main():
    if len(sys.argv) == 1:
        help()
    elif sys.argv[1] not in AVAILABLE_SCRIPTS:
        help()

    command = sys.argv[1]

    if command == 'train':
        from after_scripts import train
        sys.argv[0] = train.__name__
        app.run(train.main)
    elif command == 'prepare_dataset':
        from after_scripts import prepare_dataset
        sys.argv[0] = prepare_dataset.__name__
        app.run(prepare_dataset.main)
    elif command == 'update_dataset':
        from after_scripts import update_dataset
        sys.argv[0] = update_dataset.__name__
        app.run(update_dataset.main)
    elif command == 'train_autoencoder':
        from after_scripts import train_autoencoder
        sys.argv[0] = train_autoencoder.__name__
        app.run(train_autoencoder.main)
    elif command == 'export_autoencoder':
        from after_scripts import export_autoencoder
        sys.argv[0] = export_autoencoder.__name__
        app.run(export_autoencoder.main)
    elif command == 'export':
        from after_scripts import export
        sys.argv[0] = export.__name__
        app.run(export.main)
    elif command == 'export_midi':
        from after_scripts import export_midi
        sys.argv[0] = export_midi.__name__
        app.run(export_midi.main)
    elif command == 'export_clock':
        from after_scripts import export_clock
        sys.argv[0] = export_clock.__name__
        app.run(export_clock.main)
    else:
        raise Exception(f'Command {command} not found')
