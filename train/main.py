import train


def run(config, train_dataloader, valid_dataloader, model, save_folder):
    trainer = getattr(train, config["trainer"])(config, save_folder)
    model = trainer.train(train_dataloader, valid_dataloader, model)
    return model
