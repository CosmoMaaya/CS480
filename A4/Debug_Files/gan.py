       x_real, y_real = x.view(-1, 784).to(device), torch.full((batch_size,), 1, dtype=torch.float, device=device).unsqueeze(-1)
        discriminator_out = discriminator(x_real)
        discriminator_real_loss = nn.BCELoss()(discriminator_out, y_real)

        z = torch.randn(batch_size, latent_size).to(device)
        x_fake, y_fake = generator(z), torch.full((batch_size,), 0, dtype=torch.float, device=device).unsqueeze(-1)

        discriminator_out = discriminator(x_fake)
        discriminator_fake_loss = nn.BCELoss()(discriminator_out, y_fake)

        discriminator_loss = discriminator_real_loss + discriminator_fake_loss
        discriminator_loss.backward()
        discriminator_optimizer.step()

        generator_optimizer.zero_grad()
        new_d_out = discriminator(x_fake)
        generator_loss = nn.BCELoss()(new_d_out, y_real)
        generator_loss.backward()
        generator_optimizer.step()

        avg_generator_loss += generator_loss.item()
        avg_discriminator_loss += discriminator_loss.item()
    