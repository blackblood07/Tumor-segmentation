#initializing Dataset

class prepareDs(Dataset):
    def __init__(self, df= brain_mri_df, 
                 pix_brightness = pix_brightness, 
                 img_trans=img_trans, msk_trans=msk_trans):
        self.df = df
        self.img_trans = img_trans
        self.msk_trans = msk_trans
        self.pix_brightness= pix_brightness

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.df.loc[idx, 'img_path']
        mask_path = self.df.loc[idx, 'mask_path']

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        image, mask = self.pix_brightness(image, mask)

        if self.img_trans:
            image = self.img_trans(image).float()

        if self.msk_trans:
            mask = self.msk_trans(mask)
        return image, mask


# PyTorch Dataloaders for training validation and testing
def Data_loaders(df= brain_mri_df,
                    train_num= int(brain_mri_df.shape[0] * .6), 
                    valid_num= int(brain_mri_df.shape[0] * .8), 
                    bs = 32):
    
    train = df[:train_num].reset_index(drop=True)
    valid = df[train_num : valid_num].reset_index(drop=True)    
    test  = df[valid_num:].reset_index(drop=True)

    train_ds = prepareDs(df = train)
    valid_ds = prepareDs(df = valid)
    test_ds = prepareDs(df = test)

    train_loader = DataLoader(train_ds, batch_size = bs, shuffle = True)
    valid_loader = DataLoader(valid_ds, batch_size = bs, shuffle = False)
    test_loader = DataLoader(test_ds, batch_size = 4, shuffle = True)
    
    print("setup complete")
    
    return train_loader, valid_loader, test_loader,len(train_ds),len(test_ds)