from dataclasses import dataclass

@dataclass
class User:
    first_name: str
    last_name: str
    email_prefix: str
    email_domain: str
    honorific: str

    def __post_init__(self):
        self.email = f"{self.email_prefix}@{self.email_domain}"
        self.full_name = f"{self.first_name} {self.last_name}"
        self.honorific_full_name = f"{self.honorific} {self.full_name}"
        self.email_honorific_full_name = f"{self.honorific} {self.email_prefix}@{self.email_domain}"

    def get_names(self):
        '''
        Returns a list of all the possible ways this user can be addressed.
        '''
        return [self.first_name, 
                self.last_name, 
                self.full_name, 
                self.honorific_full_name, 
                self.email_prefix, 
                self.email, 
                self.email_honorific_full_name]
