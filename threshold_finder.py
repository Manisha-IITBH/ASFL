
from ImageClassification_Task.threshold import Threshold_Tester

if __name__=="__main__":
    trainer = Threshold_Tester()
    trainer.populate_key_value_store()
    trainer.clear_cache()
    trainer.load_best_models()
    trainer.freeze_server_back()
    trainer.calc_disc_train_thresholds()
    
    print(15*"=","GENERALISED INFERENCE",15*"=")
    trainer.inference(mode='gen')
    print(15*"=","PERSONALISED INFERENCE",15*"=")
    trainer.inference(mode='per')
    
    trainer.modify_thresh()
    
    print(15*"=","SAMPLEWISE HYBRID INFERENCE",15*"=")
    # trainer.inference_new_contrastive_samplewise()
    trainer.inference_new_contrastive_samplewise_testwise()
    print(15*"=","BATCHWISE HYBRID INFERENCE",15*"=")
    # trainer.inference_new_contrastive_batchwise()
    trainer.inference_new_contrastive_batchwise_testwise()