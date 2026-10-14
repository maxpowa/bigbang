extern crate proc_macro;
#[macro_use]
extern crate quote;
#[macro_use]
extern crate syn;

use crate::proc_macro::TokenStream;
use syn::DeriveInput;

#[proc_macro_derive(AsEntity)]
pub fn derive_as_entity(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    // type name
    let name = &input.ident;

    // generics
    let generics = input.generics;
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let expanded = quote! {
        impl #impl_generics AsEntity for #name #ty_generics #where_clause {
            fn as_entity(&self) -> bigbang::Entity {
                bigbang::Entity::new(
                    self.vx,
                    self.vy,
                    self.vz,
                    self.x,
                    self.y,
                    self.z,
                    self.radius,
                    self.mass
                )
            }
        }
    };

    TokenStream::from(expanded)
}
