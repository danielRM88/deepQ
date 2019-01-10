from player import Player

player = Player("SuperMarioBros-Nes")
player.train(True)
player.play("./models/model.ckpt")