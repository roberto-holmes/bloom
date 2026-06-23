use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, parse_macro_input};

#[proc_macro_derive(Index)]
pub fn derive_index(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);

    //use compile_error inside a quote to report errors -
    //https://stackoverflow.com/questions/54392702/how-to-report-errors-in-a-procedural-macro-using-the-quote-macro

    let Data::Struct(input_struct) = ast.data else {
        return quote! {
            compile_error!("input is not a struct type");
        }
        .into();
    };
    let mut members = input_struct.fields.iter();
    let Some(field) = members.next() else {
        return quote! {
            compile_error!("struct does not have any fields.");
        }
        .into();
    };

    //assert that the remaining fields have the same type,

    //potentially could expand this by allowing to specify a common type
    //that each field can go to via .into::<T>()
    //via a attribute macro (input.attrs)
    //#derive[Index]
    //#[base_ty=QueueIndex]
    //struct ...

    let field_ty = &field.ty;
    if !members.all(|field| &field.ty == field_ty) {
        return quote! {
            compile_error!("struct fields do not have matching types.");
        }
        .into();
    }

    let name = &ast.ident;
    let field_names: Vec<_> = input_struct.fields.iter().map(|f| &f.ident).collect();
    let field_count = input_struct.fields.iter().count();

    quote! {
        impl #name {
            pub fn len(&self) -> usize{
                #field_count
            }
        }

        impl std::ops::Index<usize> for #name {
            type Output = #field_ty;

            fn index(&self, index: usize) -> &Self::Output {
                [#(&self.#field_names),*][index]
            }
        }

        impl std::ops::IndexMut<usize> for #name {
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                //emit an array filled with refs to field names,
                //then index into it
                [#(&mut self.#field_names),*][index]
            }
        }
    }
    .into()
}

#[cfg(test)]
mod tests {
    use super::*;

    pub struct TestIndex(u32);

    #[derive(Index)]
    pub struct TestStruct {
        pub viewport: TestIndex,
        pub present: TestIndex,
        pub sync: TestIndex,
        pub ray: TestIndex,
        pub physics: TestIndex,
        pub ocean: TestIndex,
    }

    const TS: TestStruct = TestStruct {
        viewport: TestIndex(0),
        present: TestIndex(1),
        sync: TestIndex(2),
        ray: TestIndex(3),
        physics: TestIndex(4),
        ocean: TestIndex(5),
    };

    #[test]
    fn derive_index_test() {
        for i in 0u32..5u32 {
            assert!(TS.index(i as usize).0 == i);
        }
    }

    #[test]
    fn len_test() {
        assert!(TS.len() == 6);
    }
}
