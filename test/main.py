import test


def run(config, test_loader, model, save_folder):
    tester = getattr(test, config["tester"])(config, save_folder)
    tester.test(test_loader, model)
