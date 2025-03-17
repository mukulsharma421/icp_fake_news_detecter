use ic_cdk::export::candid::{candid_method, CandidType};
use ic_cdk_macros::*;
use std::cell::RefCell;

thread_local! {
    static MODEL: RefCell<Option<Vec<f64>>> = RefCell::new(None);
}

#[update]
#[candid_method(update)]
fn set_model(model_data: Vec<f64>) {
    MODEL.with(|m| *m.borrow_mut() = Some(model_data));
}

#[query]
#[candid_method(query)]
fn get_model() -> Option<Vec<f64>> {
    MODEL.with(|m| m.borrow().clone())
}